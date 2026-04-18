""" Apply SVD onto the matrix Ai """

using DelimitedFiles
using LinearAlgebra


export load_matrix_Ai
function load_matrix_Ai(file)
    Ai = readdlm(file)
    return Float64.(Ai)
end


export compute_svd
function compute_svd(Ai, k)
    
    # Now apply svd Julia builtin already
    U, S, V = svd(Ai)

    # Truncate the matrices to k
    U_k = U[:, 1:k]
    S_k = S[1:k]
    V_k = V[:, 1:k]

    S_kd = Diagonal(S_k) # Because svd return S as a vector

    return U_k, S_kd, V_k
end
