using Images
using PyPlot



# Create a gaussian filter
function makegaussianfilter(size::Array{Int,2},sigma::Float64)

  # Gaussian filter in x and y direction
  gx =  1/(sqrt(2*pi)*sigma)*[  exp(-((-1)^2/(2*sigma^2))) exp(-(0^2/(2*sigma^2))) exp(-(1^2/(2*sigma^2)))   ]
  gy =  1/(sqrt(2*pi)*sigma)*[  exp(-((-1)^2/(2*sigma^2))); exp(-(0^2/(2*sigma^2))); exp(-(1^2/(2*sigma^2))) ]

  return f::Array{Float64,2}
end

# Create a binomial filter
function makebinomialfilter(size::Array{Int,2})
  binomialWeightsx = zeros(size[1]);
  binomialWeightsy = zeros(size[2]);

  #Pascals triangle reason for the shift -1 TODO Remove

  #Calculate the binomial Weights in x direction
  for i=1:size[1]
    binomialWeightsx[i] = binomial(size[1]-1, i-1);
  end
  # Normilize Vektor
  binomialWeightsx /= sum(binomialWeightsx);

  #Calculate the binomial Weights in y direction
  for i=1:size[2]
    binomialWeightsy[i] = binomial(size[2]-1, i-1);
  end
  # Normilize Vektor
  binomialWeightsy /= sum(binomialWeightsy);

  # Multiplay the two nomilized vektors to get the filter
  f = binomialWeightsy*binomialWeightsx';
  return f::Array{Float64,2}
end

# Downsample an image by a factor of 2
function downsample2(A::Array{Float64,2})

  return D::Array{Float64,2}
end

# Upsample an image by a factor of 2
function upsample2(A::Array{Float64,2},fsize::Array{Int,2})

  return U::Array{Float64,2}
end

# Build a gaussian pyramid from an image.
# The output array should contain the pyramid levels in decreasing sizes.
function makegaussianpyramid(im::Array{Float32,2},nlevels::Int,fsize::Array{Int,2},sigma::Float64)

  return G::Array{Array{Float64,2},1}
end

# Display a given image pyramid (laplacian or gaussian)
function displaypyramid(P::Array{Array{Float64,2},1})

  return nothing
end

# Build a laplacian pyramid from a gaussian pyramid.
# The output array should contain the pyramid levels in decreasing sizes.
function makelaplacianpyramid(G::Array{Array{Float64,2},1},nlevels::Int,fsize::Array{Int,2})

  return L::Array{Array{Float64,2},1}
end

# Amplify frequencies of the first two layers of the laplacian pyramid
function amplifyhighfreq2(L::Array{Array{Float64,2},1})

  return A::Array{Array{Float64,2},1}
end

# Reconstruct an image from the laplacian pyramid
function reconstructlaplacianpyramid(L::Array{Array{Float64,2},1},fsize::Array{Int,2})

  return im::Array{Float64,2}
end


# Problem 1: Image Pyramids and Image Sharpening
#
# function problem1()
#   # parameters
#   fsize = [?, ?]
#   sigma = ?
#   nlevels = ?
#
#   # load image
#   im = PyPlot.imread("../data-julia/a2p1.png")
#
#   # create gaussian pyramid
#   G = makegaussianpyramid(im,nlevels,fsize,sigma)
#
#   # display gaussianpyramid
#   displaypyramid(G)
#   title("Gaussian Pyramid")
#
#   # create laplacian pyramid
#   L = makelaplacianpyramid(G,nlevels,fsize)
#
#   # dispaly laplacian pyramid
#   displaypyramid(L)
#   title("Laplacian Pyramid")
#
#   # amplify finest 2 subands
#   L_amp = amplifyhighfreq2(L)
#
#   # reconstruct image from laplacian pyramid
#   im_rec = reconstructlaplacianpyramid(L_amp,fsize)
#
#   # display original and reconstructed image
#   figure()
#   subplot(131)
#   imshow(im,"gray",interpolation="none")
#   axis("off")
#   title("Original Image")
#   subplot(132)
#   imshow(im_rec,"gray",interpolation="none")
#   axis("off")
#   title("Reconstructed Image")
#   subplot(133)
#   imshow(im-im_rec,"gray",interpolation="none")
#   axis("off")
#   title("Difference")
#   gcf()
#
#   return
# end
display(makebinomialfilter([3 3]))
display(sum(makebinomialfilter([3 3])))
surf(makebinomialfilter([4 40]))
