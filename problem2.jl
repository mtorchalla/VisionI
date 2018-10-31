using Images  # Basic image processing functions
using PyPlot  # Plotting and image loading
using FileIO  # Functions for loading and storing data in the ".jld2" format
using Images


# Load the image from the provided .jld2 file
function loaddata()
  data = load("imagedata.jld2", "data")
  return data::Array{Float64,2}
end


# Separate the image data into three images (one for each color channel),
# filling up all unknown values with 0
function separatechannels(data::Array{Float64,2})
  y,x = size(data) #Picture size
  #r
  r = zeros(y,x)
  for i=1:2:Int(y)-1
    for j=1:2:x-1
      r[i,j] = data[i,j]
    end
    for j=2:2:x
      r[i+1,j] = data[i+1,j]
    end
  end
  #g
  g = zeros(y,x)
  for i=1:2:Int(y)-1
    for j=2:2:x
      g[i,j] = data[i,j]
    end
  end
  #b
  b = zeros(y,x)
  for i=2:2:Int(y)
    for j=1:2:x
      b[i,j] = data[i,j]
    end
  end
  return r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2}
end


# Combine three color channels into a single image
function makeimage(r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2})
  y, x = size(r)
  image = zeros(y, x, 3)
  image[:,:,1] = r
  image[:,:,2] = g
  image[:,:,3] = b
  return image::Array{Float64,3}
end


# Interpolate missing color values using bilinear interpolation
function interpolate(r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2})
  y,x = size(r)
  #Filter Matrices
  k_r = 1/4*[0 1 0; 1 4 1; 0 0 1]
  k_g = 1/4*[1 2 1; 2 4 2; 1 2 1]
  k_b = 1/4*[1 2 1; 2 4 2; 1 2 1]

  r_new = zeros(y,x)
  g_new = zeros(y,x)
  b_new = zeros(y,x)

  r_new = imfilter(r, centered(k_r))
  g_new = imfilter(g, centered(k_g))
  b_new = imfilter(b, centered(k_b))

  # for i=2:y-1
  #   for j=2:x-1
  #     #Konvolution for R,G,B:
  #     r_new[i,j] = sum(k_r.*r[i-1:i+1, j-1:j+1])
  #     g_new[i,j] = sum(k_g.*g[i-1:i+1, j-1:j+1])
  #     b_new[i,j] = sum(k_b.*b[i-1:i+1, j-1:j+1])
  #     #Faster, not working? #TODO
  #     # if r[i,j] == 0.0
  #     #   r_new[i,j] = sum(k_r.*r[i-1:i+1, j-1:j+1])
  #     #   print(r_new[i,j])
  #     # end
  #     # #Konvolution for G:
  #     # if g[i,j] == 0.0
  #     #   g_new[i,j] = sum(k_g.*g[i-1:i+1, j-1:j+1])
  #     # end
  #     # #Konvolution for B:
  #     # if b[i,j] == 0.0
  #     #   b_new[i,j] = sum(k_b.*b[i-1:i+1, j-1:j+1])
  #     # end
  #   end
  # end
  image = makeimage(r_new,g_new,b_new)
  return image::Array{Float64,3}
end


# Display two images in a single figure window
function displayimages(img1::Array{Float64,3}, img2::Array{Float64,3})
  subplot(121)
  imshow(img1)
  subplot(122)
  imshow(img2)
end

#= Problem 2
Bayer Interpolation =#

function problem2()
  # load raw data
  data = loaddata()
  # separate data
  r,g,b = separatechannels(data)
  # merge raw pattern
  img1 = makeimage(r,g,b)
  # interpolate
  img2 = interpolate(r,g,b)
  # display images
  displayimages(img1, img2)
  return
end

problem2()
