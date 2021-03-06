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
  # Load keypoints from file
  p = load(filename)
  # Split into keypoints from Image 1 and 2
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
  D = zeros(m,n) # Create the Matrix with zeros
  # Calculate row-wise the euclideansquaredistance
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
  pairs = zeros((min(size(p1,1),size(p2,1)),4)) # Create Pairs
  # Calculate row-wise the the pairs with the minimum distance in D
  for i=1:min(size(p1,1),size(p2,1))
    pairs[i,:] = [ p1[argmin(D[:,i]),:]; p2[i,:] ]'
  end
  # Also valid:
  # for i=1:min(size(p1,1),size(p2,1))
  #   pairs[i,:] = [ p1[i,:]; p2[argmin(D[i,:]),:] ]'
  # end
  pairs = convert(Array{Int,2}, pairs) # found indices are float, convert to Int

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
  # Plot the matches in 2 Subplots, Interest-Points marked by an 'x' and a corresponding Number 'n'
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
  r_old = zeros(Int,0)
  for i=1:k
    # Pick random point out of all remaining points
    r = rand(1:size(points1,1))
    while(any(r == r_old)) # Make sure samples are not picked twice
      r = rand(1:size(points1,1))
    end
    # Save the random point and the corresponding one from the other image
    sample1[i,:] = points1[r,:]
    sample2[i,:] = points2[r,:]

    # Save the random value to not pick it agin
    r_old = vcat(r_old,r)

  end

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
  s = 0.5*maximum(points)
  # Calculate Mean in x direction
  tx = mean(points[:,1])
  # Calculate Mean in y direction
  ty = mean(points[:,2])
  # Condition Matrix T
  T = [ 1/s 0   -tx/s;
        0   1/s -ty/s;
        0   0   1     ]

  # Convert points using Common so they can be multiplied
  points = Common.cart2hom(points')
  # Condition Points U
  U = T*points
  # Re-convert points
  points = (Common.hom2cart(points))'
  # Re-convert conditioned points
  U = (Common.hom2cart(U))'
  U = U[:,1:2]

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
  # Condition Both sets of points
  U1, T1 = condition(float(points1))
  U2, T2 = condition(float(points2))
  # Create Empty Matrix A
  A = zeros(size(U1,1)*2, 9)
  # Stack Matrix A with the Linear Equation System of the Homography
  for i=1:size(U1,1)
    A[i*2-1,:] = [       0        0  0 U1[i,1] U1[i,2] 1 -U1[i,1]*U2[i,2] -U1[i,2]*U2[i,2] -U2[i,2]]
    A[i*2,:]   = [-U1[i,1] -U1[i,2] -1       0       0 0  U1[i,1]*U2[i,1]  U1[i,2]*U2[i,1]  U2[i,1]]
  end
  # Apply SVD to A, using full SVD
  U, S, V = svd(A, full=true)
  # Get conditioned Homography from the last-right singular vector from V
  H_ = reshape(V[:,end], 3,3)'
  # Un-condition the Homography
  H = inv(T2) * H_ * T1

  debug = false
  if debug
    for i=1:size(U1,1)
      a1 = H_*[U1[i,1]; U1[i,2];1]
      print("U1: ")
      println(a1 ./ a1[3])
      print("U2: ")
      println([U2[i,1]; U2[i,2];1])
    end
    for i=1:size(U1,1)
      a1 = H*[points1[i,1]; points1[i,2];1]
      print("P1: ")
      println(a1 ./ a1[3])
      print("P2: ")
      println([points2[i,1]; points2[i,2];1])
    end
  end

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
  # Apply homography to points1 and points2
  #inv(H) # This Inverese throws the Singular Matrix Exception
  points1H = Common.hom2cart(H*Common.cart2hom(points1'))'
  points2H = Common.hom2cart(H\Common.cart2hom(points2'))'
  # Vector of the Squarred differences
  d2 = zeros(size(points1,1),1)
  # Compute Squared difference between points in both directions
  for i=1:size(points1,1)
    d2[i] = norm(points1H[i,:]-points2[i,:])^2+norm(points1[i,:]-points2H[i,:])^2
  end

  @assert length(d2) == size(points1,1)
  return d2::Array{Float64,2}
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
function findinliers(distance::Array{Float64,2},thresh::Float64)
  # Find indices of inliers
  indices = findall(distance[:,1].<thresh)
  # Number of inliers
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

  # Iterate n times using a while-loop, so the increment i can be changed if a singularity occurs
  i=1
  while i<=n
    try # In case there is an Exception we re-iterate the that iteration, so we get full number of iterations with valid values
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
      i = i + 1
    catch e
      println(e)
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
  # Cumpute the full Homography from all inliers
  H = computehomography(pairs[inliers,1:2],pairs[inliers,3:4])

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
  # Empty shell for the stitched Image, as in given description
  stitched = ones(size(im1,1),700)
  # Iterate over the stitched Image shell
  for y=1:size(stitched,1)
    for x=1:size(stitched,2)
      # Compute the position of points from Image 1 in Image 2
      xy_new = H*[x;y;1]
      # Convert to Cartesian Points
      xy_new = Common.hom2cart(xy_new) #xy_new[1] = x_new, xy_new[2] = y_new
      # Check if the calculated new points are part of Image 2
      if xy_new[2]>=1 && xy_new[2]<=size(im2,1) && xy_new[1]>=1 && xy_new[1]<=size(im2,2)
        # Interpolate the searched new points from Image 2 pixels, 0.8 is for brightness adjustment(Visual) and may be changed
        stitched[y,x] = 0.8*Images.bilinear_interpolation(im2, xy_new[2], xy_new[1])
      end
    end
  end
  # Add in Image 1 to the left edge (300px) of the stitched Image (use as much of Image 1 as possible to obtain less distortion)
  stitched[1:size(im1,1),1:300] = im1[1:end,1:300];
  # Show the stiched panorama
  figure()
  # title("Stitched Image")
  imshow(stitched,"gray")
  return nothing::Nothing
end


#---------------------------------------------------------
# Problem 2: Image Stitching
#---------------------------------------------------------
function problem2()
  # SIFT Parameters
  sigma = 1.4             # standard deviation for presmoothing derivatives

  # RANSAC Parameters
  ransac_threshold = 70.0 # inlier threshold
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

  # compute number of iterations for the RANSAC algorithm
  niterations = computeransaciterations(p,k,z)

  # apply RANSAC
  bestinliers,bestpairs,bestH = ransac(pairs,ransac_threshold,niterations)#niterations)
  @printf(" # of bestinliers : %d", length(bestinliers))

  # show best matches
  showmatches(im1,im2,bestpairs)
  title("Best 4 Matches")

  # show all inliers
  showmatches(im1,im2,pairs[bestinliers,:])
  title("All Inliers")

  # stitch images and show the result
  showstitch(im1,im2,bestH)
  title("Best Four Points")

  # recompute homography with all inliers
  H = refithomography(pairs,bestinliers)
  showstitch(im1,im2,H)
  title("Refitted Homography")

  return nothing::Nothing
end
PyPlot.close("all")
problem2()
