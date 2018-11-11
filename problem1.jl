using PyPlot
using FileIO
using JLD2

# load and return the given image
function loadimage()
  str = "a1p1.png"          #Hardcoded ImageName..
  img = PyPlot.imread(str)  #Load Image using PyPlot
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
  mirrored = img[:,end:-1:1,:]    #Mirror y-coordinates
  return mirrored::Array{Float32,3}
end

# display the normal and the mirrored image in one plot
function showimages(img1::Array{Float32,3}, img2::Array{Float32,3})
  PyPlot.subplot(121) #SubPlot: 1 Row, 2 Columns, Index 1
  PyPlot.imshow(img1)
  PyPlot.subplot(122) #SubPlot: 1 Row, 2 Columns, Index 2
  PyPlot.imshow(img2)
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
