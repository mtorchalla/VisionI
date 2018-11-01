using Images
using PyPlot

# Create 3x3 derivative filters in x and y direction
function createfilters()
  sigma = 0.9
  dx = 0.5*[-1. 0. 1.]
  dy = 0.5*[-1.; 0.; 1.] #dy = dx'
  gax = [gaussF(-1, sigma) gaussF(0, sigma) gaussF(1, sigma)]
  gy = [gaussF(-1, sigma); gaussF(0, sigma); gaussF(1, sigma)] #gy = gx'
  # println(dx, dy, gx, gy)
  fx = gy * dx
  fy = dy * gax
  # fx = 0.5*[-1 0 1; -1 0 1; -1 0 1]
  # fy = [gaussY(-1, sigma); gaussY(0,sigma); gaussY(1,sigma)]
  # fy = hcat(fy, fy, fy)
  return fx::Array{Float64,2}, fy::Array{Float64,2}
end

function gaussF(a::Int64, sigma::Float64)
  return 1/(sqrt(2*pi)*sigma)*exp(-(a^2/(2*sigma^2)))::Float64
end

# Apply derivate filters to an image and return the derivative images
function filterimage(I::Array{Float32,2},fx::Array{Float64,2},fy::Array{Float64,2})
  Ix = imfilter(I, centered(fx), "replicate")
  Iy = imfilter(I, centered(fy), "replicate")
  imshow(Ix)
  return Ix::Array{Float64,2},Iy::Array{Float64,2}
end


# Apply thresholding on the gradient magnitudes to detect edges
function detectedges(Ix::Array{Float64,2},Iy::Array{Float64,2}, thr::Float64)
  grad_magn = sqrt.(Ix.^2 + Iy.^2)
  edges = grad_magn = grad_magn .* (grad_magn .> thr)
  return edges::Array{Float64,2}
end


# Apply non-maximum-suppression
function nonmaxsupp(edges::Array{Float64,2},Ix::Array{Float64,2},Iy::Array{Float64,2})

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
  # figure()
  # imshow(edges2,"gray", interpolation="none")
  # axis("off")
  # title("Non-maximum suppression")
  # gcf()
  # return
end

problem4()
