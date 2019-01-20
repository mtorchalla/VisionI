using Images
using PyPlot
using Statistics
using LinearAlgebra
using Printf

@inline isBigger(a, b) = (a > b)
@inline isSmaller(a, b) = (a < b)

include("Common.jl")

#---------------------------------------------------------
# Load ground truth disparity map
#
# Input:
#   filename:       the name of the disparity file
#
# Outputs:
#   disparity:      [h x w] ground truth disparity
#   mask:           [h x w] validity mask
#---------------------------------------------------------
function loadGTdisaprity(filename)
  disparity_gt = PyPlot.imread(filename)
  disparity_gt = Float64.(disparity_gt)*256
  mask = zeros(size(disparity_gt))
  for i=1:size(mask,1)
    for j=1:size(mask,2)
      if disparity_gt[i,j] > 0
        mask[i,j] = true
      end
    end
  end
  mask = convert(BitArray{2}, mask)

  @assert size(mask) == size(disparity_gt)
  return disparity_gt::Array{Float64,2}, mask::BitArray{2}
end

#---------------------------------------------------------
# Calculate NC between two image patches
#
# Inputs:
#   patch1 : an image patch from the left image
#   patch2 : an image patch from the right image
#
# Output:
#   nc_cost : Normalized Correlation cost
#
#---------------------------------------------------------
function computeNC(patch1, patch2)
  w_l = patch1[:] #Convert to Array
  w_r = patch2[:] #Convert to Array
  w_l_m = mean(w_l)
  w_r_m = mean(w_r)
  nc_cost = Float64.((w_l .- w_l_m)' * (w_r .- w_r_m) / ( norm(w_l .- w_l_m) * norm(w_r .- w_r_m) ))

  return nc_cost::Float64
end

#---------------------------------------------------------
# Calculate SSD between two image patches
#
# Inputs:
#   patch1 : an image patch from the left image
#   patch2 : an image patch from the right image
#
# Output:
#   ssd_cost : SSD cost
#
#---------------------------------------------------------
function computeSSD(patch1, patch2)
  ssd_cost = Float64.(sum((patch1.-patch2).^2))

  return ssd_cost::Float64
end


#---------------------------------------------------------
# Calculate the error of estimated disparity
#
# Inputs:
#   disparity : estimated disparity result, [h x w]
#   disparity_gt : ground truth disparity, [h x w]
#   valid_mask : validity mask, [h x w]
#
# Output:
#   error_disparity : calculated disparity error
#   error_map:  error map, [h x w]
#
#---------------------------------------------------------
function calculateError(disparity, disparity_gt, valid_mask)
  disparity = disparity.*valid_mask #only check valid points
  error_disparity = Float64.(norm(disparity-disparity_gt,2))
  error_map = disparity - disparity_gt

  @assert size(disparity) == size(error_map)
  return error_disparity::Float64, error_map::Array{Float64,2}
end


#---------------------------------------------------------
# Compute disparity
#
# Inputs:
#   gray_l : a gray version of the left image, [h x w]
#   gray_R : a gray version of the right image, [h x w]
#   max_disp: Maximum disparity for the search range
#   w_size: window size
#   cost_ftn: a cost function for caluclaing the cost between two patches.
#             It can be either computeSSD or computeNC.
#
# Output:
#   disparity : disparity map, [h x w]
#
#---------------------------------------------------------
function computeDisparity(gray_l, gray_r, max_disp, w_size, cost_ftn::Function)
  max_disp = Int(max_disp/2)
  disparity = zeros(Int64, size(gray_l))
  wy = Int(floor(w_size[1]/2))
  wx = Int(floor(w_size[2]/2))
  for y=1 + wy : size(gray_l,1) - wy
    for x=1 + wx : size(gray_l,2) - wx
      #Check if search range is in loop range of x: [1 + wx : size(gray_l,2) - wx]
      ix_l = x-max_disp < 1+wx ? 1+wx : x-max_disp
      ix_r = x+max_disp > size(gray_l,2) - wx ? size(gray_l,2) - wx : x+max_disp
      if cost_ftn == computeSSD #must minimize SSD
        curr_disp = Inf
      else                      #must maximize NC
        curr_disp = 0
      end
      for i=ix_l:ix_r # i := x-d
        calc_disp = cost_ftn(gray_l[y-wy:y+wy, x-wx:x+wx], gray_r[y-wy:y+wy, i-wx:i+wx])
        if cost_ftn == computeSSD #must minimize SSD
          if calc_disp < curr_disp
            curr_disp = calc_disp
            disparity[y,x] = abs(x-i) #d = |x-i|
          end
        else                      #must maximize NC
          if calc_disp > curr_disp
            curr_disp = calc_disp
            disparity[y,x] = abs(x-i) #d = |x-i|
          end
        end
      end
    end
  end
  # display(disparity)

  @assert size(disparity) == size(gray_l)
  return disparity::Array{Int64,2}
end

#---------------------------------------------------------
#   An efficient implementation
#---------------------------------------------------------
function computeDisparityEff(gray_l, gray_r, max_disp, w_size)
  # Maximum disparity we would like to check for
  max_disp = Int(max_disp/2)
  disparity = zeros(Int64, size(gray_l))
  # Boundary sizes
  wy = Int(floor(w_size[1]/2))
  wx = Int(floor(w_size[2]/2))
  # Extended right image to be able to perform a matrix convolution type action
  gray_r_ex = Float64.(Inf*ones(size(gray_r,1),size(gray_r,2)+max_disp*2))
  gray_r_ex[:,max_disp:end-max_disp-1] = gray_r
  # Squared differences for each pixel and each disparity
  sd = Float64.(Inf*ones(size(gray_l,1),size(gray_l,2),max_disp*2))
  # Sum of the squared differnences in the window size for each disparity
  sum_e = Float64.(zeros(max_disp*2))
  # Calculate the Squared differences
  for i=1:max_disp*2
    sd[:,:,i] = (gray_l.-gray_r_ex[:,i:end-(max_disp*2-i+1)]).^2
  end
  # Evaluate each pixel
  for y=1 + wy : size(gray_l,1) - wy
    for x=1 + wx : size(gray_l,2) - wx
      #Check if search range is in loop range of x: [1 + wx : size(gray_l,2) - wx]
      ix_l = x-max_disp < 1+wx ? 1+wx : x-max_disp
      ix_r = x+max_disp > size(gray_l,2) - wx ? size(gray_l,2) - wx : x+max_disp
      # Sum up the squared differnences in the window size for the range of the disparity
      for i=1:max_disp*2
        sum_e[i] = sum(sd[y-wy:y+wy,x-wx:x+wx,i])
      end
        # sum_e_sort[1:2:end] = sum_e[max_disp:-1:1]
        # sum_e_sort[2:2:end] = sum_e[max_disp+1:end]
        # Determine the disparity by cheking for the entry with the least sum of squared differnences
        disparity[y,x] = abs(argmin(sum_e)-max_disp) #d = |x-i|
    end
  end
  # display(disparity)

  @assert size(disparity) == size(gray_l)

  @assert size(disparity) == size(gray_l)
  return disparity::Array{Int64,2}
end

#---------------------------------------------------------
# Problem 2: Stereo matching
#---------------------------------------------------------
function problem2()

  # Define parameters
  w_size = [11 11]
  max_disp = 100
  gt_file_name = "a4p2_gt.png"

  # Load both images
  gray_l, rgb_l = Common.loadimage("a4p2_left.png")
  gray_r, rgb_r = Common.loadimage("a4p2_right.png")

  # Load ground truth disparity
  disparity_gt, valid_mask = loadGTdisaprity(gt_file_name)

  # estimate disparity
  @time disparity_ssd = computeDisparity(gray_l, gray_r, max_disp, w_size, computeSSD)
  @time disparity_nc = computeDisparity(gray_l, gray_r, max_disp, w_size, computeNC)


  # Calculate Error
  error_disparity_ssd, error_map_ssd = calculateError(disparity_ssd, disparity_gt, valid_mask)
  @printf(" disparity_SSD error = %f \n", error_disparity_ssd)
  error_disparity_nc, error_map_nc = calculateError(disparity_nc, disparity_gt, valid_mask)
  @printf(" disparity_NC error = %f \n", error_disparity_nc)

  figure()
  subplot(2,1,1), imshow(disparity_ssd, interpolation="none"), axis("off"), title("disparity_ssd")
  subplot(2,1,2), imshow(error_map_ssd, interpolation="none"), axis("off"), title("error_map_ssd")
  gcf()

  figure()
  subplot(2,1,1), imshow(disparity_nc, interpolation="none"), axis("off"), title("disparity_nc")
  subplot(2,1,2), imshow(error_map_nc, interpolation="none"), axis("off"), title("error_map_nc")
  gcf()

  figure()
  imshow(disparity_gt)
  axis("off")
  title("disparity_gt")
  gcf()

  @time disparity_ssd_eff = computeDisparityEff(gray_l, gray_r, max_disp, w_size)
  error_disparity_ssd_eff, error_map_ssd_eff = calculateError(disparity_ssd_eff, disparity_gt, valid_mask)
  @printf(" disparity_SSD_eff error = %f \n", error_disparity_ssd_eff)

  figure()
  subplot(2,1,1), imshow(disparity_ssd_eff, interpolation="none"), axis("off"), title("disparity_ssd_eff")
  subplot(2,1,2), imshow(error_map_ssd_eff, interpolation="none"), axis("off"), title("error_map_ssd_eff")
  gcf()

  return nothing::Nothing
end

PyPlot.close("all")
problem2()
