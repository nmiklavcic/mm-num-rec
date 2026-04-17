""" Go through matrices and compute SVD for each Ai and save it into its folder. """

using LinearAlgebra

# Dostop do svd programa:
include(joinpath(@__DIR__, "svd.jl"))

# nastavi k - to bo stevilo singularnih vrednosti
k = 50
## vecji kot je k manjsi je error na koncu, ampak vec placa zasede

input_mapa = joinpath(@__DIR__, "..", "processed_matrices")
output_mapa = joinpath(@__DIR__, "..", "svd_matrices")

for i in range(0, 9)
    println("Executing SVD for A_$i...")

    # load matrix Ai
    matrix_path = joinpath(input_mapa, "A_$i.txt")
    Ai = load_matrix_Ai(matrix_path)

    # now execute SVD on Ai
    U, S, V = compute_svd(Ai, k)

    # create folder for this numeral
    numeral_mapa = joinpath(output_mapa, "A_$i")
    isdir(numeral_mapa) || mkdir(numeral_mapa)
    writedlm(joinpath(numeral_mapa, "A_$i(U).txt"), U)
    writedlm(joinpath(numeral_mapa, "A_$i(S).txt"), S)
    writedlm(joinpath(numeral_mapa, "A_$i(V).txt"), V)
    
    Ai_remake = U * S * V'
    println("error for A_$i: ")
    println(norm(Ai-Ai_remake))
end
