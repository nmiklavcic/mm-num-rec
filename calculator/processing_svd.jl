using LinearAlgebra, Images, DelimitedFiles
include(joinpath(@__DIR__, "../processed_matrices/matrices_Ai.jl"))
include(joinpath(@__DIR__, "../pre-processing/preprocessing.jl"))

mapa_svd = joinpath(@__DIR__, "..", "svd_matrices")
include(joinpath(@__DIR__, "..", "pre-processing", "svd.jl"))


function calculate_best_fit_svd(b, k)
    BD = -1 # Best Digit 
    BR = Inf # Best Residual
    total_numerals = 10
    residuals = zeros(total_numerals)
    #XI = []

    for i in range(0, total_numerals-1)
        Ui = load_matrix_Ai(joinpath(mapa_svd, "A_$i", "A_$i(U).txt"))
        Si = load_matrix_Ai(joinpath(mapa_svd, "A_$i", "A_$i(S).txt"))

        println("Checking dimensions for i=$i")

        println("size(U$i): ", size(Ui))
        println("length(b): ", length(b))

        # --- safety check ---
        if size(Ui,1) != length(b)
            error("Dimension mismatch: U rows != b length")
        end
        
        # ci bo U Transponirano * b, desna stran enačbe
        ci = Ui' * b

        S = vec(Si) # covert Si into one vector of singular values
        r = length(S) # kolko imamo singularnih vrednosti
        corrected_k = min(k,r) # če je manj singularnih vrednosti vzami range

        if corrected_k < length(ci)
            residual = norm(ci[corrected_k:end])
        else
            residual = 0.0
        end
        
        # računamo yi
        #yi = Si \ ci

        #residual = norm(ci - Si*yi)
        residuals[i+1] = residual # dodaj ostanek v array residualov

        println("i = $i, residual = $residual")

        if residual < BR
            BR = residual
            BD = i
        end
    end
    
    # Convert residuals to probabilites so we can see how close guesses weree to one another
    INV_RES = 1.0 ./ residuals # we invert the residuals so that the best one has the highest value
    PROB = INV_RES ./ sum(INV_RES) # calculate the probability of each number based on residuals
    CONF = round(PROB[BD+1], digits=2) # the probability of the number it sees as the best fit
    #X = XI[BD+1]

    println("\nBest model: i= $BD with residual = $BR")
    return BD, CONF, PROB
    
end


function flatten_input_image(file)
    "Recieves an image, flattens it and returns the vector"
    img = load(file)
    gray = Gray.(img)
    num = find_boundries(gray)
    imr = resize_image_to_28x28(num)
    flat = mat_to_vec(imr)

    return flat
end

### Code for manual testing on input data
b = flatten_input_image(joinpath(@__DIR__, "../data/1/2026-04-14 14-46 1_crop_standardized.png"))

# determine k
k = 40 # ??

BD, CONF, PROB = calculate_best_fit_svd(b, k) #need to define k bro
PROB = round.(PROB, digits=3)
println(BD," ", CONF,"\n")
for i in 1:10
    println("$(i-1): ", PROB[i])
end

#img = x_to_img(X, BD)
###

