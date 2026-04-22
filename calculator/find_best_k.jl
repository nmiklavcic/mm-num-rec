using Images, FileIO, DelimitedFiles, LinearAlgebra, Plots

include(joinpath(@__DIR__, "processing_svd.jl"))


function mod_calculate_best_fit_svd(b, k)
    BD = -1
    BR = Inf
    total_numerals = 10
    residuals = zeros(total_numerals)

    for i in 0:total_numerals-1
        Ui = load_matrix_Ai(joinpath(@__DIR__, "..", "svd_matrices", "k_$k", "A_$i", "A_$i(U).txt"))

        if size(Ui, 1) != length(b)
            error("Dimension mismatch: size(U,1)=$(size(Ui,1)) != length(b)=$(length(b))")
        end

        # Ui is already 784×k (pre-truncated); project b onto its column space
        ci = Ui' * b
        residual = norm(b - Ui * ci)
        residuals[i+1] = residual

        if residual < BR
            BR = residual
            BD = i
        end
    end

    INV_RES = 1.0 ./ residuals
    PROB = INV_RES ./ sum(INV_RES)
    CONF = round(PROB[BD+1], digits=2)

    return BD, CONF, PROB
end

function calculate_best_k()
    test_data = joinpath(@__DIR__, "../test_data/")

    acc = []
    conf_corr = []
    conf_wr = []

    for k in 1:59
        acc_at_k        = 0
        conf_corr_sum   = 0.0
        conf_corr_count = 0
        conf_wr_sum     = 0.0
        conf_wr_count   = 0
        im_num          = 0

        for i in 0:9
            dir   = joinpath(test_data, "$i")
            files = readdir(dir)

            for file in files
                im_num += 1
                b = flatten_input_image(joinpath(dir, file))
                BD, CONF, PROB = mod_calculate_best_fit_svd(b, k)

                if BD == i
                    acc_at_k      += 1
                    conf_corr_sum += CONF
                    conf_corr_count += 1
                else
                    conf_wr_sum   += CONF
                    conf_wr_count += 1
                end
            end
        end

        append!(acc, acc_at_k / im_num)
        append!(conf_corr,conf_corr_count > 0 ? conf_corr_sum / conf_corr_count : 0.0) 
        append!(conf_wr,conf_wr_count   > 0 ? conf_wr_sum   / conf_wr_count   : 0.0)
        println("Done for k: $k")

    end

    return acc, conf_corr, conf_wr
end

function graph_res()
    acc, conf_corr, conf_wr = calculate_best_k()

    ks = 1:59

    p1 = plot(ks, acc,
        xlabel="k", ylabel="Accuracy",
        title="Recognition Accuracy vs k",
        label="accuracy", lw=2,
        marker=:circle, markersize=3,
        ylims=(0, 1))

    p2 = plot(ks, conf_corr,
        xlabel="k", ylabel="Confidence",
        title="Confidence vs k",
        label="Avg confidence for correct predictions", lw=2,
        marker=:circle, markersize=3,
        color=:green, ylims=(0, 1))
    plot!(p2, ks, conf_wr,
        label="Avg confidence for wrong predictions", lw=2,
        marker=:square, markersize=3,
        color=:red)

    p = plot(p1, p2, layout=(2, 1), size=(800, 700))
    savefig(p, joinpath(@__DIR__, "svd_stats.png"))
    display(p)

    return acc, conf_corr, conf_wr
end

graph_res()