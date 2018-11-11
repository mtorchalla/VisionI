using Images
using PyPlot

# Create 3x3 derivative filters in x and y direction
function createfilters()

  # Parameter for he Gaussian Function
  sigma = 0.9

  # Gradient filter in x and y direction
  dx = 0.5*[-1. 0. 1.]
  dy = 0.5*[-1.; 0.; 1.]

  # Gaussian filter in x and y direction
  gx =  1/(sqrt(2*pi)*sigma)*[  exp(-((-1)^2/(2*sigma^2))) exp(-(0^2/(2*sigma^2))) exp(-(1^2/(2*sigma^2)))   ]
  gy =  1/(sqrt(2*pi)*sigma)*[  exp(-((-1)^2/(2*sigma^2))); exp(-(0^2/(2*sigma^2))); exp(-(1^2/(2*sigma^2))) ]

  # Multiply the Gaussian and gradient filter to get a 3x3 filter matrix each for the x and y gardient
  fx = gy * dx
  fy = dy * gx

  return fx::Array{Float64,2}, fy::Array{Float64,2}
end

# Apply derivate filters to an image and return the derivative images
function filterimage(I::Array{Float32,2},fx::Array{Float64,2},fy::Array{Float64,2})

  # Apply the gardient filters for x and y to the image with "replicate" padding
  Ix = imfilter(I, centered(fx), "replicate")
  Iy = imfilter(I, centered(fy), "replicate")

  return Ix::Array{Float64,2},Iy::Array{Float64,2}
end


# Apply thresholding on the gradient magnitudes to detect edges
function detectedges(Ix::Array{Float64,2},Iy::Array{Float64,2}, thr::Float64)

  # Calculate the Gradient Magnitude of the combination of Ix and Iy
  grad_magn = sqrt.(Ix.^2 + Iy.^2)

  # Apply the threshold to the gradient magnitude to get a Binary image showing edges
  edges = float(grad_magn .> thr)

  return edges::Array{Float64,2}
end


# Apply non-maximum-suppression
function nonmaxsupp(edges::Array{Float64,2},Ix::Array{Float64,2},Iy::Array{Float64,2})

  # Calculate the Gradient Magnitude of the combination of Ix and Iy
  grad_magn = sqrt.(Ix.^2 + Iy.^2)

  # Non Maximum supression
  for x=2:1:size(edges)[1]-1
    for y=2:1:size(edges)[2]-1
      # Choose Edge orientation in x or y direction
      if abs(Ix[x,y]) > abs(Iy[x,y]) # Edge driection is y
        # Save the to be inspected neighbours
        inspected = [grad_magn[x+Int(sign(Ix[x,y])),y+1] grad_magn[x+Int(sign(Ix[x,y])),y-1]]
        # Check if one of the two neighbours is greater than the center pixel
        if (inspected[1]>grad_magn[x,y] || inspected[2]>grad_magn[x,y])
          # Set the center pixel to Zero
          edges[x,y] = 0.0
        end
      else # Edge driection is x
        # Save the to be inspected neighbours
        inspected = [grad_magn[x+1,y+Int(sign(Iy[x,y]))] grad_magn[x-1,y+Int(sign(Iy[x,y]))]]
        # Check if one of the two neighbours is greater than the center pixel
        if (inspected[1]>grad_magn[x,y] || inspected[2]>grad_magn[x,y])
          # Set the center pixel to Zero
          edges[x,y] = 0.0
        end
      end
    end
  end

  return edges::Array{Float64,2}
end


#= Problem 4
Image Filtering and Edge Detection =#

function problem4()

  # load image
  img = PyPlot.imread("a1p4.png")

  # create filters
  fx, fy = createfilters()

  # filter image
  imgx, imgy = filterimage(img, fx, fy)

  # show filter results
  figure()
  subplot(121)
  imshow(imgx, "gray", interpolation="none")
  title("x derivative")
  axis("off")
  subplot(122)
  imshow(imgy, "gray", interpolation="none")
  title("y derivative")
  axis("off")
  gcf()

  # show gradient magnitude
  figure()
  imshow(sqrt.(imgx.^2 + imgy.^2),"gray", interpolation="none")
  axis("off")
  title("Derivative magnitude")
  gcf()

  # threshold derivative
  threshold = 15. / 255. #threshold should be small enough, to detect the edge of the house on the surface of the water and big enough to ignore most of the reflection 
  edges = detectedges(imgx,imgy,threshold)
  figure()
  imshow(edges.>0, "gray", interpolation="none")
  axis("off")
  title("Binary edges")
  gcf()

  # non maximum suppression
  edges2 = nonmaxsupp(edges,imgx,imgy)
  figure()
  imshow(edges2,"gray", interpolation="none")
  axis("off")
  title("Non-maximum suppression")
  gcf()
  return
end

problem4()
