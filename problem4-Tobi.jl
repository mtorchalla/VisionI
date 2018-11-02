using Images
using PyPlot

# Create 3x3 derivative filters in x and y direction
function createfilters()
  #TODO Normalize Filter correctly
  sig = 0.9

  x_diff = (1/2)*[-1 0 1]
  y_gauss = [exp(-1/(2*sig^2)); exp(-0/(2*sig^2)); exp(-1/(2*sig^2))]./(sqrt(2*pi)*sig)

  y_diff = (1/2)*[-1; 0; 1]
  x_gauss = [exp(-1/(2*sig^2)) exp(-0/(2*sig^2)) exp(-1/(2*sig^2))]./(sqrt(2*pi)*sig)

  fx = y_gauss*x_diff
  fy = y_diff*x_gauss

  return fx::Array{Float64,2}, fy::Array{Float64,2}
end

# Apply derivate filters to an image and return the derivative images
function filterimage(I::Array{Float32,2},fx::Array{Float64,2},fy::Array{Float64,2})
  Ix = imfilter(I,centered(fx))
  Iy = imfilter(I,centered(fy))

  #TODO sign of the derivates
  #TODO Bounding Box

  return Ix::Array{Float64,2},Iy::Array{Float64,2}
end


# Apply thresholding on the gradient magnitudes to detect edges
function detectedges(Ix::Array{Float64,2},Iy::Array{Float64,2}, thr::Float64)

  # for x=1:size(Ix)[1]
  #   for y=1:size(Ix)[2]
  #     if Ix[x,y]<thr
  #        Ix[x,y]= 0
  #     else
  #        Ix[x,y] = 1
  #     end
  #   end
  # end
  #
  # for x=1:size(Iy)[1]
  #   for y=1:size(Iy)[2]
  #     if Iy[x,y]<thr
  #        Iy[x,y]= 0
  #     else
  #        Iy[x,y] = 1
  #     end
  #   end
  # end
  edges = zeros(size(Ix))

  for x=1:size(edges)[1]
    for y=1:size(edges)[2]

      edges[x,y] = sqrt(Ix[x,y]^2+Iy[x,y]^2)  # Gradient Magnitude

      if edges[x,y]<thr                       # Thresholding
         edges[x,y]= 0
      else
         edges[x,y] = 1
      end

    end
  end

  return edges::Array{Float64,2}
end


# Apply non-maximum-suppression
function nonmaxsupp(edges::Array{Float64,2},Ix::Array{Float64,2},Iy::Array{Float64,2})
  gradient_magnitude = zeros(size(Ix))


  # Gradient Magnitude
  for x=1:size(gradient_magnitude)[1]
    for y=1:size(gradient_magnitude)[2]
      gradient_magnitude[x,y] = sqrt(Ix[x,y]^2+Iy[x,y]^2)
    end
  end

  # Non Maximum supression
  for x=2:1:size(edges)[1]-1
    for y=2:1:size(edges)[2]-1
      if abs(Ix[x,y]) > abs(Iy[x,y])  # Choose Edge orientation
        inspected = [gradient_magnitude[x+Int(sign(Ix[x,y])),y+1] gradient_magnitude[x+Int(sign(Ix[x,y])),y-1]] # Save inspected neighbours
        if (inspected[1]>gradient_magnitude[x+Int(sign(Ix[x,y])),y] || inspected[2]>gradient_magnitude[x+Int(sign(Ix[x,y])),y]) # Check if neighbour is greater
          edges[x,y] = 0.0
        end
      else
        inspected = [gradient_magnitude[x+1,y+Int(sign(Iy[x,y]))] gradient_magnitude[x-1,y+Int(sign(Iy[x,y]))]]
        if (inspected[1]>gradient_magnitude[x,y+Int(sign(Iy[x,y]))] || inspected[2]>gradient_magnitude[x,y+Int(sign(Iy[x,y]))])
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
  threshold = 18. / 255.
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
