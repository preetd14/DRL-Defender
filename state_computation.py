# This code is derived from the RanSMAP paper: https://www.sciencedirect.com/science/article/pii/S0167404824005078?ref=pdf_download&fr=RR-2&rr=900f8853d9ad058b

import numpy as np

def calculate_storage_feature(df_read_t, df_write_t, T_window=1e8):
    """
    Calculates storage access pattern features for the given time window.

    Args:
      df_read_t (pd.DataFrame): Pandas DataFrame for read operations within the time window.
      df_write_t (pd.DataFrame): Pandas DataFrame for write operations within the time window.
      T_window (float): Time window in nanoseconds (default: 1e8, i.e., 0.1 seconds).

    Returns:
      list: A list containing the calculated storage features:
            - T_sr (float): Read throughput.
            - T_sw (float): Write throughput.
            - V_sr (float): Variance of read LBAs.
            - V_sw (float): Variance of write LBAs.
            - H_sw (float): Average Shannon entropy of write operations.
    """
    # Calculate throughput
    T_sr = df_read_t['size'].sum() / T_window if len(df_read_t) > 0 else 0
    T_sw = df_write_t['size'].sum() / T_window if len(df_write_t) > 0 else 0

    # Calculate LBA variance
    V_sr = np.var(df_read_t['LBA']) if len(df_read_t) > 1 else 0
    V_sw = np.var(df_write_t['LBA']) if len(df_write_t) > 1 else 0

    # Calculate Shannon entropy
    H_sw = df_write_t['entropy'].fillna(-1).mean() if not df_write_t.empty else -1

    # Return feature vector
    return [T_sr, T_sw, V_sr, V_sw, H_sw]

def calculate_memory_feature(df_read_t, df_write_t, df_readwrite_t, df_exec_t, T_window=1e8):
    """
    Calculates memory access pattern features for the given time window.

    Args:
      df_read_t (pd.DataFrame): Pandas DataFrame for memory read operations within the time window.
      df_write_t (pd.DataFrame): Pandas DataFrame for memory write operations within the time window.
      df_readwrite_t (pd.DataFrame): Pandas DataFrame for memory read/write operations within the time window.
      df_exec_t (pd.DataFrame): Pandas DataFrame for memory execute operations within the time window.
      T_window (float): Time window in nanoseconds (default: 1e8, i.e., 0.1 seconds).

    Returns:
      list: A list containing the calculated memory features:
            - H_mw (float): Average Shannon entropy of write operations.
            - H_mrw (float): Average Shannon entropy of read/write operations.
            - C_4KBr (int): Count of 4KB memory read operations.
            - C_4KBw (int): Count of 4KB memory write operations.
            - C_4KBrw (int): Count of 4KB memory read/write operations.
            - C_4KBx (int): Count of 4KB memory execute operations.
            - C_2MBr (int): Count of 2MB memory read operations.
            - C_2MBw (int): Count of 2MB memory write operations.
            - C_2MBrw (int): Count of 2MB memory read/write operations.
            - C_2MBx (int): Count of 2MB memory execute operations.
            - C_MMIOr (int): Count of MMIO read operations.
            - C_MMIOw (int): Count of MMIO write operations.
            - C_MMIOrw (int): Count of MMIO read/write operations.
            - C_MMIOx (int): Count of MMIO execute operations.
            - V_mr (float): Variance of read GPAs.
            - V_mw (float): Variance of write GPAs.
            - V_mrw (float): Variance of read/write GPAs.
            - V_mx (float): Variance of execute GPAs.
    """
    # Calculate Shannon entropy
    H_mw = df_write_t['entropy'].fillna(-1).mean() if not df_write_t.empty else -1
    H_mrw = df_readwrite_t['entropy'].fillna(-1).mean() if not df_readwrite_t.empty else -1

    # Count page accesses
    C_4KBr = len(df_read_t[df_read_t['type'] == 1])
    C_4KBw = len(df_write_t[df_write_t['type'] == 1])
    C_4KBrw = len(df_readwrite_t[df_readwrite_t['type'] == 1])
    C_4KBx = len(df_exec_t[df_exec_t['type'] == 1])

    C_2MBr = len(df_read_t[df_read_t['type'] == 2])
    C_2MBw = len(df_write_t[df_write_t['type'] == 2])
    C_2MBrw = len(df_readwrite_t[df_readwrite_t['type'] == 2])
    C_2MBx = len(df_exec_t[df_exec_t['type'] == 2])

    C_MMIOr = len(df_read_t[df_read_t['type'] == 4])
    C_MMIOw = len(df_write_t[df_write_t['type'] == 4])
    C_MMIOrw = len(df_readwrite_t[df_readwrite_t['type'] == 4])
    C_MMIOx = len(df_exec_t[df_exec_t['type'] == 4])

    # Calculate GPA variance
    V_mr = np.var(df_read_t['GPA']) if len(df_read_t) > 1 else 0
    V_mw = np.var(df_write_t['GPA']) if len(df_write_t) > 1 else 0
    V_mrw = np.var(df_readwrite_t['GPA']) if len(df_readwrite_t) > 1 else 0
    V_mx = np.var(df_exec_t['GPA']) if len(df_exec_t) > 1 else 0

    # Return feature vector
    return [
        H_mw, H_mrw, C_4KBr, C_4KBw, C_4KBrw, C_4KBx, C_2MBr, C_2MBw,
        C_2MBrw, C_2MBx, C_MMIOr, C_MMIOw, C_MMIOrw, C_MMIOx, V_mr,
        V_mw, V_mrw, V_mx
    ]

def compute_state(df_ata_read, df_ata_write, df_mem_read, df_mem_write, df_mem_readwrite, df_mem_exec, t, T_window):
    """
    Calculates the state vector by combining storage and memory access pattern features
    for the given time window.

    Args:
      df_ata_read (pd.DataFrame): Pandas DataFrame for ATA read operations.
      df_ata_write (pd.DataFrame): Pandas DataFrame for ATA write operations.
      df_mem_read (pd.DataFrame): Pandas DataFrame for memory read operations.
      df_mem_write (pd.DataFrame): Pandas DataFrame for memory write operations.
      df_mem_readwrite (pd.DataFrame): Pandas DataFrame for memory read/write operations.
      df_mem_exec (pd.DataFrame): Pandas DataFrame for memory execute operations.
      t (float): Starting time of the time window in nanoseconds.
      T_window (float): Time window in nanoseconds.

    Returns:
      np.ndarray: The calculated state vector as a NumPy array.
    """
    # Extract data within the time window
    df_ata_read_t = df_ata_read[(df_ata_read['ts'] >= t) & (df_ata_read['ts'] < t + T_window)]
    df_ata_write_t = df_ata_write[(df_ata_write['ts'] >= t) & (df_ata_write['ts'] < t + T_window)]
    df_mem_read_t = df_mem_read[(df_mem_read['ts'] >= t) & (df_mem_read['ts'] < t + T_window)]
    df_mem_write_t = df_mem_write[(df_mem_write['ts'] >= t) & (df_mem_write['ts'] < t + T_window)]
    df_mem_readwrite_t = df_mem_readwrite[(df_mem_readwrite['ts'] >= t) & (df_mem_readwrite['ts'] < t + T_window)]
    df_mem_exec_t = df_mem_exec[(df_mem_exec['ts'] >= t) & (df_mem_exec['ts'] < t + T_window)]

    # Calculate storage and memory features
    storage_features = calculate_storage_feature(df_ata_read_t, df_ata_write_t, T_window=T_window)
    memory_features = calculate_memory_feature(df_mem_read_t, df_mem_write_t, df_mem_readwrite_t, df_mem_exec_t, T_window=T_window)

    # Combine and flatten the features
    state = np.concatenate((storage_features, memory_features))
    return state