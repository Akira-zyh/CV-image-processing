using PyCall # A library which allows Julia call Python's library
using TyImages # A library for processing image
cv2 = pyimport("cv2") # import opencv from Python

function extract(image)
    """ Description: Extract lane line from image            

        Argument: 
            image: Image for processing
        
        return:
            resultBin(bool metrix): Processed Binary mask

    """
    sThrld = (170, 255)
    ## Saturation channel threshold (from 170 to 255): Keep pixels whose saturation is high and filter the others out
    sblThrld = (40, 200)
    ## Sobel edge detect result threshold (from 195 to 200): 

    hls = float(cv2.cvtColor(image, cv2.COLOR_BGR2HLS)) # Transform color space into HLS
    # Channel order of HLS: [H, L, S]
    lChannel = hls[:, :, 2] # L Channel
    sChannel = hls[:, :, 3] # S Channel

    sBin = (sChannel .> sThrld[1]) .& (sChannel .<= sThrld[2])
    # Filter parts whose saturation is between 170 and 255. Generate bool maskcode

    sobelx = cv2.Sobel(lChannel, cv2.CV_64F, 1, 0) # Horizontal Sobel edge detect of L channel
    abs_sobelx = abs.(sobelx) # get ABS
    normSobelx = 255 * abs_sobelx ./ maximum(abs_sobelx) # normalization to 0~255

    normSobelx = round.(normSobelx)
    scaled_sobel = convert.(UInt8, normSobelx)

    hSblBin = (scaled_sobel .>= sblThrld[1]) .& (scaled_sobel .<= sblThrld[2])
    # Filter edge parts whose S channel values are between 195 and 200 and convert into UInt8

    lBin = lChannel .> 100
    # File parts whose L channel values are greater than 100

    resultBin = (hSblBin .| sBin) .& lBin
    # Combine results of Sobel edge detect, Saturation and light threshold to generate binary mask

    return resultBin
end


cap = cv2.VideoCapture("data/project_video.mp4") # Capture the frame
width = Int(cap.get(3)) # get video width value
height = Int(cap.get(4)) # get video height value
fps = cap.get(5) # get FPS value
path = "data/" # set storage path for "output.mp4"

# Initialize writter by using HP4V encoder
fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
# Attention!!!---VideoWritter_fourcc function of this version of Julia probably does not support pass the argument like [*'mp4v']
out = cv2.VideoWriter("./data/output.mp4", fourcc, fps, (width, height), isColor=true)


fnum = cap.get(7) # get the total frame quanity of the video
for i = 1:20 
    (ret, frame) = cap.read() # capture the frame 
    if !ret
        println("Can't receive frame(stream end?).Exiting...")
        break
    end
    frame = extract(frame) # extract the lane line 
    frame = frame .* 255 
    frame = cv2.merge((frame, frame, frame)) # Extend the binary mask
    frame = UInt8.(frame) # Convert bool into UInt8
    out.write(frame) # write frames into output.mp4

    # Optional: generate jpg
    filename = path * string(lpad(i, 4, "0"), ".jpg")
    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
end

# release the resources
cap.release()
out.release()
cv2.destroyAllWindows()