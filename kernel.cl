#define BS 64 // block size : local work set size
#define ITEMS 16  //

__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  const int CBS = BS / ITEMS;                               // interval of column
  const int localRow  = get_local_id(0);                    // local row [0, BS)
  const int localCol  = get_local_id(1);                    // local col [0, CBS)
  const int globalRow = get_group_id(0) * BS + localRow;    // global row [0, M)
  const int globalCol = get_group_id(1) * BS + localCol;    // global col [0, N)        

  // set local block(BS x BS) of A, B in local memory
  __local float A_block[BS][BS];
  __local float B_block[BS][BS];

  // C[globalRow][globalCol + i * CBS] = ret[i]
  float ret[ITEMS];
  for (int i = 0; i < ITEMS; i++) {
    ret[i] = 0.0f;
  }
  
  const int numBlocks = (K + BS - 1) / BS;
  for (int b = 0; b < numBlocks; b++) {
    const int r = BS * b + localRow;
    const int c = BS * b + localCol;

    // Load a block of A and B into local memory
    for (int i = 0; i < ITEMS; i++) {
      if (globalRow < M && c + i * CBS < K)
        A_block[localRow][localCol + i * CBS] = A[globalRow * K + c + i * CBS];
      else
        A_block[localRow][localCol + i * CBS] = 0;
      if (r < K && globalCol + i * CBS < N)
        B_block[localRow][localCol + i * CBS] = B[r * N + globalCol + i * CBS];
      else 
        B_block[localRow][localCol + i * CBS] = 0;
    }   

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < BS; k++) {
      for (int i = 0; i < ITEMS; i++) {
        ret[i] += A_block[localRow][k] * B_block[k][localCol + i * CBS];
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Store the results in C
  for(int i = 0; i < ITEMS; i++) {
    if(globalRow < M && globalCol + i * CBS < N) 
      C[globalRow * N + globalCol + i * CBS] = ret[i];
  }
}
