using PyPlot
using FileIO
using JLD2
using Images

# load and return the given image
function loadimage()
  str = "a1p1.png"  #Hardcoded ImageName..
  img = load(str)   #Load Image with FileIO
  img = float(channelview(img)) #using Images, convert RGB repr. to 3D-Array
  return img::Array{Float32,3}
end

# save the image as a .jld2 file
function savefile(img::Array{Float32,3})
  @save "img.jld2" img  #using JLD2 macro, save img Array
end

# load and return the .jld2 file
function loadfile()
  @load "img.jld2" img  #using JLD2 macro, load img Array
  return img::Array{Float32,3}
end

# create and return a horizontally mirrored image
function mirrorhorizontal(img::Array{Float32,3})
  mirrored = img[:,:,end:-1:1]
  return mirrored::Array{Float32,3}
end

# display the normal and the mirrored image in one plot
function showimages(img1::Array{Float32,3}, img2::Array{Float32,3})
  #swap rgb channel to the end
  rgb,x,y = size(img1)
  img1_temp = zeros(x, y, rgb)
  for i=1:3 #format for imgshow: [x,y,RGB]
    img1_temp[:,:,i] = img1[i,:,:]
  end
  rgb,x,y = size(img2)
  img2_temp = zeros(x, y, rgb)
  for i=1:3 #format for imgshow: [x,y,RGB]
    img2_temp[:,:,i] = img2[i,:,:]
  end
  subplot(121)
  imshow(img1_temp)
  subplot(122)
  imshow(img2_temp)
end

#= Problem 1
Load and Display =#

function problem1()
  img1 = loadimage()
  savefile(img1)
  img2 = loadfile()
  img2 = mirrorhorizontal(img2)
  showimages(img1, img2)
end

problem1()
