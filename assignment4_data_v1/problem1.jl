using PyPlot
using FileIO
using Statistics
using LinearAlgebra

include("Common.jl")

#---------------------------------------------------------
# Conditioning: Normalization of coordinates for numeric stability.
#
# INPUTS:
#   points    unnormalized coordinates
#
# OUTPUTS:
#   U         normalized (conditioned) coordinates
#   T         [3x3] transformation matrix that is used for
#                   conditioning
#
#---------------------------------------------------------
function condition(points::Array{Float64,2})
  #points = points[1:2,:]'
  # Calculate Maximum of the Points
  s = 0.5*maximum(points)
  # Calculate Mean in x direction
  tx = mean(points[1,:])
  # Calculate Mean in y direction
  ty = mean(points[2,:])
  # Condition Matrix T
  T = [ 1/s 0   -tx/s;
        0   1/s -ty/s;
        0   0   1     ]
  # Condition Points U
  U = T*points

  @assert size(U) == size(points)
  @assert size(T) == (3,3)
  return U::Array{Float64,2},T::Array{Float64,2}
end


#---------------------------------------------------------
# Enforce a rank of 2 to a given 3x3 matrix.
#
# INPUTS:
#   A     [3x3] matrix (of rank 3)
#
# OUTPUTS:
#   Ahat  [3x3] matrix of rank 2
#
#---------------------------------------------------------
# Enforce that the given matrix has rank 2
function enforcerank2(A::Array{Float64,2})
  # Apply SVD to the Fundamental Matrix
  U,S,V = svd(A,full=true)
  # Enforce rank two on the Fundamental Matix
  Ahat = U*[S[1] 0 0;0 S[2] 0; 0 0 0]*V'

  @assert size(Ahat) == (3,3)
  return Ahat::Array{Float64,2}
end


#---------------------------------------------------------
# Compute fundamental matrix from conditioned coordinates.
#
# INPUTS:
#   p1     set of conditioned coordinates in left image
#   p2     set of conditioned coordinates in right image
#
# OUTPUTS:
#   F      estimated [3x3] fundamental matrix
#
#---------------------------------------------------------
# Compute the fundamental matrix for given conditioned points
function computefundamental(p1::Array{Float64,2},p2::Array{Float64,2})

  A=zeros(size(p1,2),9)
  # Create the linear equation system to estimate the Fundamental Matrix
  for i=1:size(p1,2)

    x1 = p1[1,i]
    x2 = p2[1,i]
    y1 = p1[2,i]
    y2 = p2[2,i]

    A[i,:]= kron(p1[:,i]',p2[:,i]')#[x2*x1 y2*x1 x1 x2*y1 y2*y1 y1 x2 y2 1]
  end
  # Apply SVD to estimate a Solution
  U,S,V = svd(A,full=true)
  # Take the last singular Vector as the solution for the Fundamental Matrix
  F = [V[1:3,end]';V[4:6,end]';V[7:9,end]']
  # Enforce Rank two on the previously estimated Fundamental Matrix
  F = enforcerank2(F)

  @assert size(F) == (3,3)
  return F::Array{Float64,2}
end


#---------------------------------------------------------
# Compute fundamental matrix from unconditioned coordinates.
#
# INPUTS:
#   p1     set of unconditioned coordinates in left image
#   p2     set of unconditioned coordinates in right image
#
# OUTPUTS:
#   F      estimated [3x3] fundamental matrix
#
#---------------------------------------------------------
function eightpoint(p1::Array{Float64,2},p2::Array{Float64,2})
  # Condition the Points for better Numerical stability
  x1,T1 = condition(p1)
  x2,T2 = condition(p2)
  # Compute the Fundamental Matrix
  F = computefundamental(x1,x2)
  # Un-Condition the Fundamental Matrix
  F = T1'*F*T2

  @assert size(F) == (3,3)
  return F::Array{Float64,2}
end


#---------------------------------------------------------
# Draw epipolar lines:
#   E.g. for a given fundamental matrix and points in first image,
#   draw corresponding epipolar lines in second image.
#
#
# INPUTS:
#   Either:
#     F         [3x3] fundamental matrix
#     points    set of coordinates in left image
#     img       right image to be drawn on
#
#   Or:
#     F         [3x3] transposed fundamental matrix
#     points    set of coordinates in right image
#     img       left image to be drawn on
#
#---------------------------------------------------------
function showepipolar(F::Array{Float64,2},points::Array{Float64,2},img::Array{Float64,3})
  # Calculate the Epipole
  e = nullspace(F)
  # Normalize the Epipole coordinates
  e = e./e[3]
  # Convert Points to homogeneous coordinates
  x = Common.cart2hom(points')
  # Calculate the Vectors of the Epipolar Lines
  l = (F')*x

  imshow(img,interpolation="none")
  # Plot the epipoles
  PyPlot.plot(e[1],e[2],"x")
  # Plot the Epipolar Lines with overestimated range
  for i=1:size(points,1)
    l[1:2,i] = l[1:2,i]/l[3,i]
    l[1:2,i] = l[1:2,i]/norm(l[1:2,i])
    PyPlot.plot([e[1];e[1]+3000*l[2,i]],[e[2];e[2]-3000*l[1,i]],"black",linewidth=0.7)
    PyPlot.plot([e[1];e[1]-3000*l[2,i]],[e[2];e[2]+3000*l[1,i]],"black",linewidth=0.7)
  end
  # Limit the shown graph to the original Image
  PyPlot.xlim(0,size(img,2))
  PyPlot.ylim(size(img,1),0)
  PyPlot.show()
  return nothing::Nothing
end


#---------------------------------------------------------
# Compute the residual errors for a given fundamental matrix F,
# and set of corresponding points.
#
# INPUTS:
#    p1    corresponding points in left image
#    p2    corresponding points in right image
#    F     [3x3] fundamental matrix
#
# OUTPUTS:
#   residuals      residual errors for given fundamental matrix
#
#---------------------------------------------------------
function computeresidual(p1::Array{Float64,2},p2::Array{Float64,2},F::Array{Float64,2})
  residual = zeros(size(p1,2),1)
  # Calculate the remaining Error of the Fundamental Martix with the Residuals
  for i=1:size(p1,2)
    residual[i] = p1[:,i]'*F*p2[:,i]
  end

  return residual::Array{Float64,2}
end


#---------------------------------------------------------
# Problem 1: Fundamental Matrix
#---------------------------------------------------------
function problem1()
  # Load images and points
  img1 = Float64.(PyPlot.imread("a4p1a.png"))
  img2 = Float64.(PyPlot.imread("a4p1b.png"))
  points1 = load("points.jld2", "points1")
  points2 = load("points.jld2", "points2")

  # Display images and correspondences
  figure()
  subplot(121)
  imshow(img1,interpolation="none")
  axis("off")
  scatter(points1[:,1],points1[:,2])
  title("Keypoints in left image")
  subplot(122)
  imshow(img2,interpolation="none")
  axis("off")
  scatter(points2[:,1],points2[:,2])
  title("Keypoints in right image")

  # compute fundamental matrix with homogeneous coordinates
  x1 = Common.cart2hom(points1')
  x2 = Common.cart2hom(points2')
  F = eightpoint(x1,x2)

  # draw epipolar lines
  figure()
  subplot(121)
  F_transposed = permutedims(F, [2, 1])
  showepipolar(F_transposed,points2,img1)
  scatter(points1[:,1],points1[:,2])
  title("Epipolar lines in left image")
  subplot(122)
  showepipolar(F,points1,img2)
  scatter(points2[:,1],points2[:,2])
  title("Epipolar lines in right image")

  # check epipolar constraint by computing the remaining residuals
  residual = computeresidual(x1, x2, F)
  println("Residuals:")
  println(residual)

  # compute epipoles
  U,_,V = svd(F)
  e1 = V[1:2,3]./V[3,3]
  println("Epipole 1: $(e1)")
  e2 = U[1:2,3]./U[3,3]
  println("Epipole 2: $(e2)")

  return
end
PyPlot.close("all")
problem1()
