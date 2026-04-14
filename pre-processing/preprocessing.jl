using Images, FileIO, DelimitedFiles

# println("Image size: ", size(gray))
# println("Top-left corner: ", gray[1, 1])
# println("Center: ", gray[140, 140])
# println("Min value in image: ", minimum(gray))
# println("Max value in image: ", maximum(gray))

# for i in 1:280
#     println(gray[i, :])
# end

## Turns image form 280×280 to 28×28 and turns it into a matrix with white 0 and black 1
# gray = 1 .- Gray.(imresize(img, (28,28)))


## Find number boundries 
## We are searching for the left most and right most columns that contain non-zero values.
## As well as the top most and bottom most rows that contain non-zero values.

function find_boundries(img)
    "It is a function that recieves 280×280 matrix and finds the columns and rows that bind the number and returns them"
    
    "The image is a perfect 280×280 squre, so we just move through them till we find the first non zero value."
    "We since the image is small we can do this using linear search"
    "If we want to optimize we can use binary search"

    img = 1 .- img


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

        if sum(img[i, :]) > 0 && TMP_L == 0
            TMP = i
            TMP_L = 1
        end
        if sum(img[281-i, :]) > 0 && BMP_L == 0
            BMP = 281 - i
            BMP_L = 1
        end
        if sum(img[:, i]) > 0 && LMP_L == 0
            LMP = i
            LMP_L = 1
        end
        if sum(img[:, 281-i]) > 0 && RMP_L == 0
            RMP = 281 - i
            RMP_L = 1
        end
    end

    num = img[TMP:BMP, LMP:RMP]
    
    return num
end


function resize_image_to_28x28(num)
    "Function recieves an image of unknown dimensions and resizes it to 280 by 280 so that the image it recieved is in the middle"

    H, W = size(num)
    
    ## DEBUG
    # println(H," ",W)
    ###

    MH = 280 - H # Missing height
    MW = 280 - W # Missing width

    PAD_TB = Int(floor(MH / 2)) # Padding top bottom 
    PAD_LR = Int(floor(MW / 2)) # Padding left right

    canvas = fill(Gray(0.0), 280, 280)
    canvas[PAD_TB+1 : PAD_TB + H, PAD_LR+1 : PAD_LR+W] = num # Places image in to the middle of new matrix 280×280
    
    ## DEBUG
    # for i in 1:280
    #     println(canvas[i, : ])
    # end
    ###
    
    imr = imresize(canvas, (28,28))

    return imr
end

function mat_to_vec(imr)
    "Function to flatten the image matrix in to a vector adn return it"
    flat = vec(Float64.(imr))
    return flat
end



function gen_matrices_Ai()
    "Function to generate matrix Ai for each i 0-9 and save them"
    
    # for loop to itterate through files without needing their name
    for i in 0:9

        files = readdir("data/$i", join=true)
        
        columns = []

        for file in files
            img = load(file)
            gray = Gray.(img)
            num = find_boundries(gray)
            imr = resize_image_to_28x28(num)
            flat = mat_to_vec(imr)
            
            push!(columns,flat)
        end

        Ai = reduce(hcat, columns) # Stack the vectors in to their respective matrix
        writedlm("processed_matrices/A_$i.txt", Ai) # Write the matrix in its respective file to be used later
    end 
end

gen_matrices_Ai()