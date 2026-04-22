using Images, FileIO, LinearAlgebra, Plots

include(joinpath(@__DIR__, "../processed_matrices/matrices_Ai.jl"))
include(joinpath(@__DIR__, "../pre-processing/preprocessing.jl"))

function flatten_input_image(file)
    img = load(file)
    gray = Gray.(img)
    num = find_boundries(gray)
    imr = resize_image_to_28x28(num)
    return mat_to_vec(imr)
end

function calculate_best_fit(b)
    A = [A0, A1, A2, A3, A4, A5, A6, A7, A8, A9]
    residuals = zeros(10)

    for i in 0:9
        xi = A[i+1] \ b
        residuals[i+1] = norm(b - A[i+1] * xi)
    end

    BD = argmin(residuals) - 1
    INV_RES = 1.0 ./ residuals
    PROB = INV_RES ./ sum(INV_RES)
    CONF = PROB[BD+1]

    return BD, CONF, PROB
end

function calculate_accuracy_mp()
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
            BD, CONF, PROB = calculate_best_fit(b)

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

    overall_acc    = total_correct / total_count
    per_digit_acc  = per_digit_corr ./ per_digit_count
    avg_conf_corr  = conf_corr_count > 0 ? conf_corr_sum / conf_corr_count : 0.0
    avg_conf_wr    = conf_wr_count   > 0 ? conf_wr_sum   / conf_wr_count   : 0.0

    return overall_acc, per_digit_acc, avg_conf_corr, avg_conf_wr
end

function graph_res_mp()
    overall_acc, per_digit_acc, avg_conf_corr, avg_conf_wr = calculate_accuracy_mp()

    digits = 0:9

    p1 = bar(digits, per_digit_acc,
        xlabel="Digit", ylabel="Accuracy",
        title="Per-digit Accuracy (MP method)\nOverall: $(round(overall_acc, digits=3))",
        label="accuracy", ylims=(0, 1),
        xticks=0:9)
    hline!(p1, [overall_acc],
        label="overall avg", lw=2,
        color=:red, linestyle=:dash)

    p2 = bar(["Correct", "Wrong"], [avg_conf_corr, avg_conf_wr],
        xlabel="Prediction type", ylabel="Avg Confidence",
        title="Average Confidence (MP method)",
        label=false, ylims=(0, 1),
        color=[:green, :red])

    p = plot(p1, p2, layout=(1, 2), size=(900, 450))
    savefig(p, joinpath(@__DIR__, "mp_stats.png"))
    display(p)

    return overall_acc, per_digit_acc, avg_conf_corr, avg_conf_wr
end

function graph_confusion_3_mp()
    test_data = joinpath(@__DIR__, "../test_data/")
    dir   = joinpath(test_data, "3")
    files = readdir(dir)

    predictions = zeros(Int, 10)

    for file in files
        b = flatten_input_image(joinpath(dir, file))
        BD, CONF, PROB = calculate_best_fit(b)
        predictions[BD+1] += 1
    end

    total     = length(files)
    pred_frac = predictions ./ total
    colors    = [i == 3 ? :green : :red for i in 0:9]

    p = bar(0:9, pred_frac,
        xlabel="Predicted digit", ylabel="Fraction of digit 3 images",
        title="Confusion: digit 3 (MP method)\nCorrect=$(predictions[4])/$total",
        label=false, ylims=(0, 1),
        xticks=0:9,
        color=colors)

    savefig(p, joinpath(@__DIR__, "..", "documentation_of_work","mp_confusion_3.png"))
    display(p)

    return predictions
end

graph_confusion_3_mp()
