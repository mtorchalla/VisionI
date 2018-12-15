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

  ddx = [0 1 0]'*[1. -2. 1.];
  ddy = [1.; -2.; 1.]*[0 1 0];

  # Apply filters to image
  I_xx = imfilter(imfilter(img, centered(gauss), "replicate"), centered(ddx), "replicate" );
  I_yy = imfilter(imfilter(img, centered(gauss), "replicate"), centered(ddy), "replicate" );

  I_xy = imfilter(imfilter( imfilter(img, centered(gauss), "replicate") , centered(dx), "replicate" ),centered(dy),"replicate");

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
  # Hessian Determinant
  criterion = (I_xx.*I_yy-I_xy.^2).*(sigma^4)
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
  # Empty Interest Points map
  i_points = zeros(size(criterion))
  # Search range for local maxima 5x5
  fRange=0:4
  # Iterate through the image starting with the 1,1 component of the search matrix at image coordinate 1,1
  for i=1:1:size(criterion)[1]-4
    for j=1:1:size(criterion)[2]-4
        # When criterion is maximum of its 5x5 surrounding, save it to i_points
        if(criterion[i+2,j+2]==maximum(criterion[(fRange).+i,(fRange).+j]))
          i_points[i+2,j+2] = criterion[i+2,j+2]
        end

    end
  end
  # Apply threshold to the interest points
  i_points = i_points.>thresh
  # Disregard the 5 pixel outer edge of the image
  i_points[1:5,:] .= 0
  i_points[end-4:end,:] .= 0
  i_points[:,1:5] .= 0
  i_points[:,end-4:end] .= 0
  # Find all thresholded interest points by row and column
  rows,columns = Common.findnonzero(i_points)

  return rows::Array{Int,1},columns::Array{Int,1}
end


#---------------------------------------------------------
# Problem 1: Interest point detector
#---------------------------------------------------------
function problem1()
  # parameters
  sigma = 4.5              # std for presmoothing image
  fsize = 25              # filter size for smoothing
  threshold = 10^-3           # Corner criterion threshold

  # Load both colored and grayscale image from PNG file
  gray,rgb = loadimage("a3p1.png")

  # Compute the three components of the Hessian matrix
  I_xx,I_yy,I_xy = computehessian(gray,sigma,fsize)

  # Compute Hessian based corner criterion
  criterion = computecriterion(I_xx,I_yy,I_xy,sigma)

  # Display Hessian criterion image
  figure()
  imshow(criterion,"jet",interpolation="none")
  axis("off")
  title("Determinant of Hessian")
  gcf()

  # Threshold corner criterion
  mask = criterion .> threshold
  rows, columns = Common.findnonzero(mask)
  figure()
  imshow(rgb)
  plot(columns.-1,rows.-1,"xy",linewidth=8)
  axis("off")
  title("Hessian interest points without non-maximum suppression")
  gcf()

  # Apply non-maximum suppression
  rows,columns = nonmaxsupp(criterion,threshold)

  # Display interest points on top of color image
  figure()
  imshow(rgb)
  plot(columns.-1,rows.-1,"xy",linewidth=8)
  axis("off")
  title("Hessian interest points after non-maximum suppression")
  gcf()
  return nothing
end
PyPlot.close("all")
problem1()
