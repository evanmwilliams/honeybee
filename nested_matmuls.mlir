#spmm = {
  indexing_maps = [
    affine_map<(i,j,k) -> (i,k)>, // A
    affine_map<(i,j,k) -> (k,j)>, // B
    affine_map<(i,j,k) -> (i,j)>  // X (out)
  ],
  iterator_types = ["parallel", "reduction", "parallel"],
  doc = "X(i,j) += A(i,k) * B(k,j)"
}

#SPARSE = #sparse_tensor.encoding<{
  map = (i,j) -> (i : compressed, j : compressed)
}>

module {
func.func @kernel_spmm(%arga: tensor<128x128xf64, #SPARSE>,
                       %argb: tensor<128x128xf64>,
                       %argc: tensor<128x128xf64, #SPARSE>,
                       %argd: tensor<128x128xf64, #SPARSE>,
                       %arge: tensor<128x128xf64, #SPARSE>,
                       %argf: tensor<128x128xf64, #SPARSE>,
                       %argx: tensor<128x128xf64, #SPARSE>) -> tensor<128x128xf64, #SPARSE> {
    %0 = linalg.matmul ins(%arga, %argb: tensor<128x128xf64, #SPARSE>, tensor<128x128xf64>)
                       outs(%argx: tensor<128x128xf64, #SPARSE>) -> tensor<128x128xf64, #SPARSE>
    %1 = linalg.matmul ins(%0, %argc: tensor<128x128xf64, #SPARSE>, tensor<128x128xf64, #SPARSE>)
                       outs(%argx: tensor<128x128xf64, #SPARSE>) -> tensor<128x128xf64, #SPARSE>
    %2 = linalg.matmul ins(%1, %argd: tensor<128x128xf64, #SPARSE>, tensor<128x128xf64, #SPARSE>)
                       outs(%argx: tensor<128x128xf64, #SPARSE>) -> tensor<128x128xf64, #SPARSE>
    %3 = linalg.matmul ins(%2, %arge: tensor<128x128xf64, #SPARSE>, tensor<128x128xf64, #SPARSE>)
                       outs(%argx: tensor<128x128xf64, #SPARSE>) -> tensor<128x128xf64, #SPARSE>
    %4 = linalg.matmul ins(%3, %argf: tensor<128x128xf64, #SPARSE>, tensor<128x128xf64, #SPARSE>)
                       outs(%argx: tensor<128x128xf64, #SPARSE>) -> tensor<128x128xf64, #SPARSE>
    return %1 : tensor<128x128xf64, #SPARSE>
}

}