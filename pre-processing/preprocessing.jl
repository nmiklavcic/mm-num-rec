using Images, FileIO

img = load("data/1/1118513771131960.png")

gray = Gray.(img)

println("Image size: ", size(gray))
println("Top-left corner: ", gray[1, 1])
println("Center: ", gray[140, 140])
println("Min value in image: ", minimum(gray))
println("Max value in image: ", maximum(gray))


# for i in 1:280
#     println(gray[i, :])
# end

## Find number boundries 
## We are searching for the left most and right most columns that contain non-zero values.
## As well as the top most and bottom most rows that contain non-zero values.

function find_boundries(img)
    "It is a function that recieves 280×280 matrix and finds the columns and rows that bind the number and returns them"
    
    "The image is a perfect 280×280 squre, so we just move through them till we find the first non zero value."
    "We since the image is small we can do this using linear search"
    "If we want to optimize we can use binary search"

    LMP = 1 # Left most point
    RMP = 280 # Right most point
    TMP = 1 # Top most point
    BMP = 280 # Bottom most point
    LMP_L = 0 # Left most point lock
    RMP_L = 0 # Right most point lock
    TMP_L = 0 # Top most point lock 
    BMP_L = 0 # Bottom most point lock

    for i in 1:280
        ## DEBUG
        # println(sum(img[i, :]), " ", sum(img[281-i, :]), " ", sum(img[:, i]), " ", sum(img[:, 281-i]))
        ###

        if sum(img[i, :]) < 280 && TMP_L == 0
            TMP = i
            TMP_L = 1
        end
        if sum(img[281-i, :]) < 280 && BMP_L == 0
            BMP = 281 - i
            BMP_L = 1
        end
        if sum(img[:, i]) < 280 && LMP_L == 0
            LMP = i
            LMP_L = 1
        end
        if sum(img[:, 281-i]) < 280 && RMP_L == 0
            RMP = 281 - i
            RMP_L = 1
        end
    end
    return LMP, RMP, TMP, BMP
end

LMP, RMP, TMP, BMP = find_boundries(gray)
println("Left most point: ", LMP)
println("Right most point: ", RMP)
println("Top most point: ", TMP)
println("Bottom most point: ", BMP) 

for i in TMP:BMP
    println(gray[i, LMP:RMP])
end