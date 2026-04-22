using Plots

include(joinpath(@__DIR__, "find_best_k.jl"))

function calculate_accuracy_svd_at_k(k)
    test_data = joinpath(@__DIR__, "../test_data/")

    total_correct   = 0
    total_count     = 0
    per_digit_corr  = zeros(Int, 10)
    per_digit_count = zeros(Int, 10)
    conf_corr_sum   = 0.0
    conf_corr_count = 0
    conf_wr_sum     = 0.0
    conf_wr_count   = 0

    for i in 0:9
        dir   = joinpath(test_data, "$i")
        files = readdir(dir)

        for file in files
            b = flatten_input_image(joinpath(dir, file))
            BD, CONF, PROB = mod_calculate_best_fit_svd(b, k)

            per_digit_count[i+1] += 1
            total_count += 1

            if BD == i
                total_correct += 1
                per_digit_corr[i+1] += 1
                conf_corr_sum += CONF
                conf_corr_count += 1
            else
                conf_wr_sum += CONF
                conf_wr_count += 1
            end
        end
    end

    overall_acc   = total_correct / total_count
    per_digit_acc = per_digit_corr ./ per_digit_count
    avg_conf_corr = conf_corr_count > 0 ? conf_corr_sum / conf_corr_count : 0.0
    avg_conf_wr   = conf_wr_count   > 0 ? conf_wr_sum   / conf_wr_count   : 0.0

    return overall_acc, per_digit_acc, avg_conf_corr, avg_conf_wr
end

function graph_res_k_sweep()
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
    savefig(p, joinpath(@__DIR__, "svd_sweep_stats.png"))
    display(p)

    return acc, conf_corr, conf_wr
end

function graph_res_at_k(k=10)
    overall_acc, per_digit_acc, avg_conf_corr, avg_conf_wr = calculate_accuracy_svd_at_k(k)

    p = bar(0:9, per_digit_acc,
        xlabel="Digit", ylabel="Accuracy",
        title="Per-digit Accuracy (SVD k=$k)\nOverall: $(round(overall_acc, digits=3))",
        label="accuracy", ylims=(0, 1),
        xticks=0:9)
    hline!(p, [overall_acc],
        label="overall avg", lw=2,
        color=:red, linestyle=:dash)

    savefig(p, joinpath(@__DIR__, "..","documentation_of_work","svd_at_k$(k)_stats.png"))
    display(p)

    return overall_acc, per_digit_acc
end

graph_res_at_k(59)
