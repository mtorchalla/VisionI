using Images
using PyPlot
using Printf
using Random
using Statistics
using LinearAlgebra
using Interpolations

include("Common.jl")


#---------------------------------------------------------
# Loads keypoints from JLD2 container.
#
# INPUTS:
#   filename     JLD2 container filename
#
# OUTPUTS:
#   keypoints1   [n x 2] keypoint locations (of left image)
#   keypoints2   [n x 2] keypoint locations (of right image)
#
#---------------------------------------------------------
function loadkeypoints(filename::String)
  p = load(filename)
  keypoints1 = p["keypoints1"]
  keypoints2 = p["keypoints2"]

  @assert size(keypoints1,2) == 2
  @assert size(keypoints2,2) == 2
  return keypoints1::Array{Int64,2}, keypoints2::Array{Int64,2}
end


#---------------------------------------------------------
# Compute pairwise Euclidean square distance for all pairs.
#
# INPUTS:
#   features1     [128 x m] descriptors of first image
#   features2     [128 x n] descriptors of second image
#
# OUTPUTS:
#   D             [m x n] distance matrix
#
#---------------------------------------------------------
function euclideansquaredist(features1::Array{Float64,2},features2::Array{Float64,2})
  m, n = size(features1,2), size(features2,2)
  D = zeros(m,n)
  for i=1:m
    D[i,:] = sum((features1[:,i] .- features2[:,:]).^2, dims=1)
  end
  # for i=1:m
  #   for j=1:n
  #     D[i,j] = sum((features1[:,i] .- features2[:,j]).^2, dims=1)[1]
  #   end
  # end

  @assert size(D) == (size(features1,2),size(features2,2))
  return D::Array{Float64,2}
end


#---------------------------------------------------------
# Find pairs of corresponding interest points given the
# distance matrix.
#
# INPUTS:
#   p1      [m x 2] keypoint coordinates in first image.
#   p2      [n x 2] keypoint coordinates in second image.
#   D       [m x n] distance matrix
#
# OUTPUTS:
#   pairs   [min(N,M) x 4] vector s.t. each row holds
#           the coordinates of an interest point in p1 and p2.
#
#---------------------------------------------------------
function findmatches(p1::Array{Int,2},p2::Array{Int,2},D::Array{Float64,2})
  pairs = zeros((min(size(p1,1),size(p2,1)),4))
  for i=1:min(size(p1,1),size(p2,1))
    pairs[i,:] = [ p1[argmin(D[:,i]),:]; p2[i,:] ]'
  end
  pairs = convert(Array{Int,2}, pairs)

  @assert size(pairs) == (min(size(p1,1),size(p2,1)),4)
  return pairs::Array{Int,2}
end


#---------------------------------------------------------
# Show given matches on top of the images in a single figure.
# Concatenate the images into a single array.
#
# INPUTS:
#   im1     first grayscale image
#   im2     second grayscale image
#   pairs   [n x 4] vector of coordinates containing the
#           matching pairs.
#
#---------------------------------------------------------
function showmatches(im1::Array{Float64,2},im2::Array{Float64,2},pairs::Array{Int,2})
  figure()
  subplot(121)
  imshow(im1, "gray")
  for i=1:size(pairs,1)
    PyPlot.plot(pairs[i,1], pairs[i,2], "x")
    PyPlot.text(pairs[i,1], pairs[i,2], i)
  end
  subplot(122)
  imshow(im2, "gray")
  for i=1:size(pairs,1)
    PyPlot.plot(pairs[i,3], pairs[i,4], "x")
    PyPlot.text(pairs[i,3], pairs[i,4], i)
  end
  return nothing::Nothing
end


#---------------------------------------------------------
# Computes the required number of iterations for RANSAC.
#
# INPUTS:
#   p    probability that any given correspondence is valid
#   k    number of samples drawn per iteration
#   z    total probability of success after all iterations
#
# OUTPUTS:
#   n   minimum number of required iterations
#
#---------------------------------------------------------
function computeransaciterations(p::Float64,k::Int,z::Float64)
  # Calculate the Iterations from probabilities and sample size
  n = ( log2(1-z) ) / ( log2(1-p^k) )
  # Round up, to get atleast n iterations, and convert to Int
  n = Int(ceil(n))
  # print(n)
  return n::Int
end


#---------------------------------------------------------
# Randomly select k corresponding point pairs.
#
# INPUTS:
#   points1    given points in first image
#   points2    given points in second image
#   k          number of pairs to select
#
# OUTPUTS:
#   sample1    selected [kx2] pair in left image
#   sample2    selected [kx2] pair in right image
#
#---------------------------------------------------------
function picksamples(points1::Array{Int,2},points2::Array{Int,2},k::Int)
  sample1 = zeros(Int,k,2)
  sample2 = zeros(Int,k,2)
  r_old = zeros(Int,1)
  for i=1:k
    # Pick random point out of all remaining points
    r = rand(1:size(points1,1))
    while(any(r == r_old)) # Make sure smples are not picked twice
      r = rand(1:size(points1,1))
    end
    # Save the random point and the corresponding one from the other image
    sample1[i,:] = points1[r,:]
    sample2[i,:] = points2[r,:]

    # Save the random value to not pick it agin
    r_old = vcat(r_old,r)
  end
  # subplot(122)
  # PyPlot.scatter(sample2[:,1],sample2[:,2])
  @assert size(sample1) == (k,2)
  @assert size(sample2) == (k,2)
  return sample1::Array{Int,2},sample2::Array{Int,2}
end


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
  # Calculate Maximum of the Points
  s = 0.5*maximum(sqrt.(points[:,1].^2+points[:,2].^2))
  # Calculate Mean in x direction
  tx = mean(points[:,1])
  # Calculate Mean in y direction
  ty = mean(points[:,2])

  T = [ 1/s 0   -tx/s ;
        0   1/s -ty/s;
        0   0   1     ]
  # display(T)
  points = Common.cart2hom(points')

  U = T*points
  points = (Common.hom2cart(points))'
  # display(points)
  U = (Common.hom2cart(U))'
  U = U[:,1:2]
  # display(U)

  @assert size(U) == size(points)
  @assert size(T) == (3,3)
  return U::Array{Float64,2},T::Array{Float64,2}
end


#---------------------------------------------------------
# Estimates the homography from the given correspondences.
#
# INPUTS:
#   points1    correspondences in left image
#   points2    correspondences in right image
#
# OUTPUTS:
#   H         [3x3] estimated homography
#
#---------------------------------------------------------
function computehomography(points1::Array{Int,2}, points2::Array{Int,2})
  # Condition the corresponding points
  points1,T1 = condition(Float64.(points1))
  points2,T2 = condition(Float64.(points2))

  x = points1[1,1]
  y = points1[1,2]
  x1 = points2[1,1]
  y1 = points2[1,2]
  # Save first corresponding points in the Matrix A
  A = [ 0   0   0   x   y   1   -x*y1   -y*y1   -y1;
        -x  -y  -1  0   0   0    x*x1    y*x1    x1 ]
  #     Append the other Points to A
  for i = 2:size(points,1)
    x = points1[i,1]
    y = points1[i,2]
    x1 = points2[i,1]
    y1 = points2[i,2]
    B = [ 0   0   0   x   y   1   -x*y1   -y*y1   -y1;
          -x  -y  -1  0   0   0    x*x1    y*x1    x1 ]
    A = vcat(A,B)
  end
  # Calculate SVD from A
  U, S, V = svd(A,full=true)
  # display(V[:,end])
  # Take last Right singular Vector to construct H
  H = [V[1:3,end]'; V[4:6,end]'; V[7:9,end]']
  # display(H)
  # Recondition the Homography
  # display(T1)
  # display(inv(T2))
  H = T2\H*T1
  display(H)
  display(det(H))

  # display(sqrt(sum(H.^2)))

  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end


#---------------------------------------------------------
# Computes distances for keypoints after transformation
# with the given homography.
#
# INPUTS:
#   H          [3x3] homography
#   points1    correspondences in left image
#   points2    correspondences in right image
#
# OUTPUTS:
#   d2         distance measure using the given homography
#
#---------------------------------------------------------
function computehomographydistance(H::Array{Float64,2},points1::Array{Int,2},points2::Array{Int,2})
  # Apply homography to points1
  inv(H)
  points1H = Common.hom2cart(H*Common.cart2hom(points1'))'
  points2H = Common.hom2cart(H\Common.cart2hom(points2'))'
  # Compute Squarred difference between points in both directions
  d2 = (points1H[:,1]-points2[:,1]).^2+(points1H[:,2]-points2[:,2]).^2+(points1[:,1]-points2H[:,1]).^2+(points1[:,2]-points2H[:,2]).^2


  @assert length(d2) == size(points1,1)
  return d2::Array{Float64,1}#2}
end


#---------------------------------------------------------
# Compute the inliers for a given distances and threshold.
#
# INPUTS:
#   distance   homography distances
#   thresh     threshold to decide whether a distance is an inlier
#
# OUTPUTS:
#  n          number of inliers
#  indices    indices (in distance) of inliers
#
#---------------------------------------------------------
function findinliers(distance::Array{Float64,1},thresh::Float64) #2}
  indices = findall(distance.<thresh)
  n=size(indices,1)

  return n::Int,indices::Array{Int,1}
end


#---------------------------------------------------------
# RANSAC algorithm.
#
# INPUTS:
#   pairs     potential matches between interest points.
#   thresh    threshold to decide whether a homography distance is an inlier
#   n         maximum number of RANSAC iterations
#
# OUTPUTS:
#   bestinliers   [n x 1 ] indices of best inliers observed during RANSAC
#
#   bestpairs     [4x4] set of best pairs observed during RANSAC
#                 i.e. 4 x [x1 y1 x2 y2]
#
#   bestH         [3x3] best homography observed during RANSAC
#
#---------------------------------------------------------
function ransac(pairs::Array{Int,2},thresh::Float64,n::Int)
  number_old = -1
  bestinliers = 0
  bestpairs = 0
  bestH = 0

  # Iterate n times
  for i = 1:n
    # Pick random samples
    samples1, samples2 = picksamples(pairs[:,1:2],pairs[:,3:4],4)
    # Compute the Homography for the random samples
    H = computehomography(samples1,samples2)
    # Compute the distances between tranformed points
    d2 = computehomographydistance(H,pairs[:,1:2],pairs[:,3:4])
    # Find the inliers
    number, indices = findinliers(d2,thresh)
    # If the number of inliers is greater the before save them
    if(number>=number_old)
      bestinliers = indices
      bestpairs = [samples1 samples2]
      bestH = H
      number_old = number
    end
  end

  @assert size(bestinliers,2) == 1
  @assert size(bestpairs) == (4,4)
  @assert size(bestH) == (3,3)
  return bestinliers::Array{Int,1},bestpairs::Array{Int,2},bestH::Array{Float64,2}
end


#---------------------------------------------------------
# Recompute the homography based on all inliers
#
# INPUTS:
#   pairs     pairs of keypoints
#   inliers   inlier indices.
#
# OUTPUTS:
#   H         refitted homography using the inliers
#
#---------------------------------------------------------
function refithomography(pairs::Array{Int64,2}, inliers::Array{Int64,1})




  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end


#---------------------------------------------------------
# Show panorama stitch of both images using the given homography.
#
# INPUTS:
#   im1     first grayscale image
#   im2     second grayscale image
#   H       [3x3] estimated homography between im1 and im2
#
#---------------------------------------------------------
function showstitch(im1::Array{Float64,2},im2::Array{Float64,2},H::Array{Float64,2})



  return nothing::Nothing
end


#---------------------------------------------------------
# Problem 2: Image Stitching
#---------------------------------------------------------
function problem2()
  # SIFT Parameters
  sigma = 1.4             # standard deviation for presmoothing derivatives

  # RANSAC Parameters
  ransac_threshold = 50.0 # inlier threshold
  p = 0.5                 # probability that any given correspondence is valid
  k = 4                   # number of samples drawn per iteration
  z = 0.99                # total probability of success after all iterations

  # load images
  im1 = PyPlot.imread("a3p2a.png")
  im2 = PyPlot.imread("a3p2b.png")

  # Convert to double precision
  im1 = Float64.(im1)
  im2 = Float64.(im2)

  # load keypoints
  keypoints1, keypoints2 = loadkeypoints("keypoints.jld2")

  # extract SIFT features for the keypoints
  features1 = Common.sift(keypoints1,im1,sigma)
  features2 = Common.sift(keypoints2,im2,sigma)

  # compute chi-square distance  matrix
  D = euclideansquaredist(features1,features2)

  # find matching pairs
  pairs = findmatches(keypoints1,keypoints2,D)

  # show matches
  showmatches(im1,im2,pairs)
  title("Putative Matching Pairs")
  #
  # # compute number of iterations for the RANSAC algorithm
  niterations = computeransaciterations(p,k,z)
  #
  # # apply RANSAC
  bestinliers,bestpairs,bestH = ransac(pairs,ransac_threshold,1000)#niterations)
  @printf(" # of bestinliers : %d", length(bestinliers))
  #
  # # show best matches
  showmatches(im1,im2,bestpairs)
  title("Best 4 Matches")
  #
  # # show all inliers
  showmatches(im1,im2,pairs[bestinliers,:])
  title("All Inliers")
  #
  # # stitch images and show the result
  # showstitch(im1,im2,bestH)
  #
  # # recompute homography with all inliers
  # H = refithomography(pairs,bestinliers)
  # showstitch(im1,im2,H)

  return nothing::Nothing
end
PyPlot.close("all")
problem2()
