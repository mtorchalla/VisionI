using Images
using PyPlot

include("Common.jl")
#---------------------------------------------------------
# Loads grayscale and color image given PNG filename.
#
# INPUTS:
#   filename     given PNG image file
#
# OUTPUTS:
#   gray         single precision grayscale image
#   rgb          single precision color image
#
#---------------------------------------------------------
function loadimage(filename)
  rgb = float64.(PyPlot.imread(filename));
  gray = Common.rgb2gray(rgb);
  return gray::Array{Float64,2}, rgb::Array{Float64,3}
end


#---------------------------------------------------------
# Computes entries of Hessian matrix.
#
# INPUTS:
#   img             grayscale color image
#   sigma           std for presmoothing image
#   fsize           filter size for smoothing
#
# OUTPUTS:
#   I_xx       second derivative in x-direction
#   I_yy       second derivative in y-direction
#   I_xy       derivative in x- and y-direction
#
#---------------------------------------------------------
function computehessian(img::Array{Float64,2},sigma::Float64,fsize::Int)
  #  Gaussian Filter
  gauss = Common.gauss2d(sigma,[fsize fsize]);

  #  Derivative Filters
  dx = (1/6)*ones(3)*[-1. .0 1.];
  dy = (1/6)*[-1.; .0; 1.]*ones(3)';
  # dx = [1. .0 -2. .0 1.]
  # dy = [1.; .0; -2.; .0; 1.]
  # ddx = (1/8)*ones(5)*[1. .0 -2. .0 1.];
  # ddy = (1/8)*[1.; .0; -2.; .0; 1.]*ones(5)';
  ddx = [0 1 0]'*[1. -2. 1.];
  ddy = [1.; -2.; 1.]*[0 1 0];
  # ddx = imfilter(dx,centered(dx));
  # ddy = imfilter(dy,centered(dy));
  # dxy = imfilter(dx,centered(dy));

  I_xx = imfilter(imfilter(img, centered(gauss), "replicate"), centered(ddx), "replicate" );
  I_yy = imfilter(imfilter(img, centered(gauss), "replicate"), centered(ddy), "replicate" );

  # I_xx = imfilter( imfilter( imfilter(img, centered(gauss), "replicate") , centered(dx), "replicate" ), centered(dx),"replicate");
  # I_yy = imfilter(imfilter( imfilter(img, centered(gauss), "replicate") , centered(dy), "replicate" ),centered(dy),"replicate");
  # I_xx = I_x.*I_x
  # I_yy = I_y.*I_y
  # I_xy = imfilter( imfilter(img, centered(gauss), "replicate"), centered(dxy), "replicate" );
  # I_x = imfilter( imfilter(img, centered(gauss), "replicate"), centered(d1x), "replicate" );
  # I_y = imfilter( imfilter(img, centered(gauss), "replicate"), centered(d1y), "replicate" );
  # I_xy = imfilter(imfilter(imfilter(img,centered(gauss),"replicate"),centered(dx),"replicate"),centered(dy),"replicate")
  I_xy = imfilter(imfilter( imfilter(img, centered(gauss), "replicate") , centered(dx), "replicate" ),centered(dy),"replicate");
  # I_xy = imfilter(I_xy, centered(d1y), "replicate" );

  # figure()
  # subplot(221)
  # imshow(I_xx,"gray")
  # subplot(222)
  # imshow(I_yy,"gray")
  # subplot(223)
  # imshow(I_xy,"gray")
  return I_xx::Array{Float64,2},I_yy::Array{Float64,2},I_xy::Array{Float64,2}
end


#---------------------------------------------------------
# Computes function values of Hessian criterion.
#
# INPUTS:
#   I_xx       second derivative in x-direction
#   I_yy       second derivative in y-direction
#   I_xy       derivative in x- and y-direction
#   sigma      std that was used for smoothing image
#
# OUTPUTS:
#   criterion  function score
#
#---------------------------------------------------------
function computecriterion(I_xx::Array{Float64,2},I_yy::Array{Float64,2},I_xy::Array{Float64,2}, sigma::Float64)
  criterion = (I_xx.*I_yy-I_xy.^2).*sigma^4
  return criterion::Array{Float64,2}
end


#---------------------------------------------------------
# Non-maximum suppression of criterion function values.
#   Extracts local maxima within a 5x5 window and
#   allows multiple points with equal values within the same window.
#   Discards interest points in a 5 pixel boundary.
#   Applies thresholding with the given threshold.
#
# INPUTS:
#   criterion  function score
#   thresh     param for thresholding
#
# OUTPUTS:
#   rows        row positions of kept interest points
#   columns     column positions of kept interest points
#
#---------------------------------------------------------
function nonmaxsupp(criterion::Array{Float64,2}, thresh::Float64)
  # figure()
  # subplot(121)
  # imshow(criterion.>thresh,"gray")
  # title("before nonMax")
  i_points = zeros(size(criterion))
  fRange=0:4
  for i=1:1:size(criterion)[1]-4
    for j=1:1:size(criterion)[2]-4

       # mask=ones(5,5)
       #
       # mask[findall(criterion[(fRange).+i,(fRange).+j].<maximum(criterion[(fRange).+i,(fRange).+j]))].=0
       #
       # i_points[(fRange).+i,(fRange).+j] = criterion[(fRange).+i,(fRange).+j].*mask #i_points[(fRange).+i,(fRange).+j].*(-mask.+1)+
       if(criterion[i+2,j+2]==maximum(criterion[(fRange).+i,(fRange).+j]))
          i_points[i+2,j+2] = criterion[i+2,j+2]
       end

    end
  end
  # subplot(122)
  # imshow(i_points,"gray")
  i_points = i_points.>thresh

  # title("After nonMax")

  i_points[1:5,:] .= 0
  i_points[end-4:end,:] .= 0
  i_points[:,1:5] .= 0
  i_points[:,end-4:end] .= 0

  rows,columns = Common.findnonzero(i_points)

  return rows::Array{Int,1},columns::Array{Int,1}
end


#---------------------------------------------------------
# Problem 1: Interest point detector
#---------------------------------------------------------
function problem1(sigma)
  # parameters
  sigma = sigma              # std for presmoothing image
  fsize = 25              # filter size for smoothing
  threshold = 10^-3           # Corner criterion threshold

  # Load both colored and grayscale image from PNG file
  gray,rgb = loadimage("a3p1.png")

  # Compute the three components of the Hessian matrix
  I_xx,I_yy,I_xy = computehessian(gray,sigma,fsize)

  # Compute Hessian based corner criterion
  criterion = computecriterion(I_xx,I_yy,I_xy,sigma)

  # Display Hessian criterion image
  # figure()
  # imshow(criterion,"gray",interpolation="none")
  # axis("off")
  # title("Determinant of Hessian")
  # gcf()

  # Threshold corner criterion
  mask = criterion .> threshold
  rows, columns = Common.findnonzero(mask)
  # figure()
  # imshow(rgb)
  # plot(columns.-1,rows.-1,"xy",linewidth=8)
  # axis("off")
  # title("Hessian interest points without non-maximum suppression")
  # gcf()

  # Apply non-maximum suppression
  rows,columns = nonmaxsupp(criterion,threshold)

  # Display interest points on top of color image
  figure()
  imshow(imfilter(gray,centered(Common.gauss2d(sigma,[fsize fsize]))),"gray")
  plot(columns.-1,rows.-1,"xy",linewidth=8)
  axis("off")
  title("Hessian interest points after non-maximum suppression $(sigma)")
  gcf()
  return nothing
end
PyPlot.close("all")
problem1(4.5)
problem1(3.5)
problem1(2.5)
problem1(1.5)
problem1(5.5)
