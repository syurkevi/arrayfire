/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

kernel void extract_diag(global T *oPtr, const KParam out,
                         global const T *values, global const int *rowIdx,
                         global const int *colIdx, const int nRows, const int nCols) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);

    if (row >= out.dims[0]) return;

    const int rowStartIdx = rowIdx[row];
    const int rowEndIdx   = rowIdx[row + 1];

    // Repeat loop until all values in the row are computed
    for (int idx = rowStartIdx + get_local_id(0); idx < rowEndIdx;
        idx += get_local_size(0)) {
        const int col = colIdx[idx];

        //TODO: change
        if (row >= nRows || col >= nCols) continue;  // Bad indices

        // Get Values only if along diagonal
        if(col == row) {
            const T val  = values[idx];

            oPtr[row] = val;
        }
    }
}

kernel void inv_diag(global T *oPtr, const KParam out,
                     global const T *values, global const int *rowIdx,
                     global const int *colIdx, const int nRows, const int nCols) {
    const int row = get_group_id(0) * get_local_size(1) + get_local_id(1);

    if (row >= out.dims[0]) return;

    const int rowStartIdx = rowIdx[row];
    const int rowEndIdx   = rowIdx[row + 1];

    // Repeat loop until all values in the row are computed
    for (int idx = rowStartIdx + get_local_id(0); idx < rowEndIdx;
        idx += get_local_size(0)) {
        const int col = colIdx[idx];

        //TODO: change
        if (row >= nRows || col >= nCols) continue;  // Bad indices

        // Get Values only if along diagonal
        if(col == row) {
            const T val  = values[idx];

            oPtr[row] = (val > 0) ? val : (T)ONE / val;
        }
    }
}
