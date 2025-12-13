"""
Sparse matrix utilities for efficient computation
"""
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def pivot_sparse(df, index, columns, values, fill_value=None, dropna_idxcol=True, as_pd=True):
    """
    Uses scipy.sparse.coo_matrix to construct a pivot table.
    This uses less memory and is faster when the resulting pivot_table will be sparse.
    Aggregates by sum.

    Arguments:
        df (pandas dataframe)
        index (string or list): name(s) of column(s) for index
        columns (string or list): name(s) of column(s) for columns
        values (string): name of column for values
        fill_value (scalar, default None): value to replace missing values
        dropna_idxcol (bool, default True): drop rows with NaN in index/columns
        as_pd (bool, default True): return pandas dataframe or raw sparse matrix

    Returns:
        pivot_df (pandas dataframe) OR (coo, idx_labels, col_labels)
    """
    if isinstance(index, str):
        if dropna_idxcol:
            df = df.dropna(subset=[index])
    if isinstance(index, list):
        if dropna_idxcol:
            df = df.dropna(subset=index)
        if len(index) == 1:
            index = index[0]
    if isinstance(columns, str):
        if dropna_idxcol:
            df = df.dropna(subset=[columns])
    if isinstance(columns, list):
        if dropna_idxcol:
            df = df.dropna(subset=columns)
        if len(columns) == 1:
            columns = columns[0]

    if isinstance(index, str):
        idx_arr, idx_arr_unique = df[index].factorize(sort=True, use_na_sentinel=False)
    else:
        idx_arr, idx_arr_unique = pd.MultiIndex.from_frame(df[index]).factorize(sort=True, use_na_sentinel=False)
        idx_arr_unique = pd.MultiIndex.from_tuples(idx_arr_unique, names=index)
    
    if isinstance(columns, str):
        col_arr, col_arr_unique = df[columns].factorize(sort=True, use_na_sentinel=False)
    else:
        col_arr, col_arr_unique = pd.MultiIndex.from_frame(df[columns]).factorize(sort=True, use_na_sentinel=False)
        col_arr_unique = pd.MultiIndex.from_tuples(col_arr_unique, names=columns)

    coo = coo_matrix((df[values], (idx_arr, col_arr)), shape=(idx_arr_unique.shape[0], col_arr_unique.shape[0]))

    if as_pd:
        pivot_df = pd.DataFrame.sparse.from_spmatrix(coo, index=idx_arr_unique, columns=col_arr_unique)
        pivot_df.index.rename(index, inplace=True)
        pivot_df.columns.rename(columns, inplace=True)

        if fill_value is not None and fill_value is not np.nan:
            pivot_df = pivot_df.fillna(fill_value)

        return pivot_df
    else:
        return coo, idx_arr_unique, col_arr_unique


def create_sparse_interaction_matrix(ratings_df, user_col='userId', item_col='movieId', rating_col='rating'):
    """
    Create sparse user-item interaction matrix efficiently using COO format
    
    Returns:
        csr_matrix: Sparse interaction matrix
        user_id_map: Dict mapping userId to row index
        item_id_map: Dict mapping movieId to column index
        idx_to_user: Inverse mapping
        idx_to_item: Inverse mapping
    """
    print("Creating sparse interaction matrix...")
    
    # Get COO matrix and mappings
    coo, user_ids, item_ids = pivot_sparse(
        ratings_df, 
        index=user_col, 
        columns=item_col, 
        values=rating_col,
        as_pd=False
    )
    
    # Convert to CSR for efficient row operations
    csr = coo.tocsr()
    
    # Create mappings
    user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    item_id_map = {iid: idx for idx, iid in enumerate(item_ids)}
    idx_to_user = {idx: uid for uid, idx in user_id_map.items()}
    idx_to_item = {idx: iid for iid, idx in item_id_map.items()}
    
    print(f"  Shape: {csr.shape}")
    print(f"  Density: {csr.nnz / (csr.shape[0] * csr.shape[1]):.4%}")
    
    return csr, user_id_map, item_id_map, idx_to_user, idx_to_item


def sparse_cosine_similarity_top_k(matrix, k=50, batch_size=1000):
    """
    Compute top-k cosine similarities efficiently in batches
    Returns sparse matrix with only top-k neighbors per row
    
    Args:
        matrix: Input sparse matrix (users x items or items x users)
        k: Number of top neighbors to keep
        batch_size: Batch size for processing
    
    Returns:
        Sparse CSR matrix with similarity scores
    """
    from sklearn.preprocessing import normalize
    
    print(f"Computing top-{k} similarities in batches...")
    n = matrix.shape[0]
    
    # Normalize rows for cosine similarity
    matrix_norm = normalize(matrix, norm='l2', axis=1)
    
    # Store results
    rows = []
    cols = []
    data = []
    
    # Process in batches
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        
        if start % 10000 == 0:
            print(f"  Processing {start}/{n}...")
        
        # Compute similarities for this batch
        batch_sim = matrix_norm[start:end] @ matrix_norm.T
        
        # Convert to dense for top-k selection (only for batch)
        batch_sim_dense = batch_sim.toarray()
        
        # Find top-k indices for each row
        for i, sim_row in enumerate(batch_sim_dense):
            global_i = start + i
            # Exclude self-similarity
            sim_row[global_i] = -1
            
            # Get top-k indices
            top_k_idx = np.argpartition(sim_row, -k)[-k:]
            top_k_idx = top_k_idx[np.argsort(sim_row[top_k_idx])[::-1]]
            
            # Store only positive similarities
            for j in top_k_idx:
                if sim_row[j] > 0:
                    rows.append(global_i)
                    cols.append(j)
                    data.append(sim_row[j])
    
    # Create sparse matrix
    similarity_matrix = csr_matrix(
        (data, (rows, cols)), 
        shape=(n, n)
    )
    
    print(f"  Similarity matrix: {similarity_matrix.nnz:,} non-zero entries")
    
    return similarity_matrix
