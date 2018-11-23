using Images
using PyPlot



# Create a gaussian filter
function makegaussianfilter(size::Array{Int,2},sigma::Float64)
  # Gaussian filter in x and y direction
  # X direction
  gaussx = collect(1:size[2]).-(round(size[2]/2,RoundUp))

  # If the filter size is Even,everything must be shifted by -0.5 to have 0 a Center maximum
  if size[2]%2==0
    gaussx = gaussx.-0.5;
  end

  for i=1:size[2]
    gaussx[i] = exp(-((gaussx[i]^2)/(2*sigma^2)))
  end


  # Y direction
  gaussy = collect(1:size[1]).-(round(size[1]/2,RoundUp))

  # If the filter size is Even,everything must be shifted by -0.5 to have 0 a Center maximum
  if size[1]%2==0
    gaussy = gaussy.-0.5;
  end

  for i=1:size[1]
    gaussy[i] = exp(-(gaussy[i]^2)/(2*sigma^2))
  end

  # Calculate the filter Matrix and Normalize it
  gaussF = gaussy*gaussx';
  gaussF = gaussF/sum(gaussF)
  f = gaussF;

  return f::Array{Float64,2}
end

# Create a binomial filter
function makebinomialfilter(size::Array{Int,2})
  binomialWeightsx = zeros(size[2]);
  binomialWeightsy = zeros(size[1]);



  #Calculate the binomial Weights in x direction
  for i=1:size[2]
    binomialWeightsx[i] = binomial(size[2]-1, i-1); # Shift -1 is used because the the number of binomal coefficeints is n+1 and k runs from 0 to n
  end

  # Normilize Vektor
  binomialWeightsx /= sum(binomialWeightsx);

  #Calculate the binomial Weights in y direction
  for i=1:size[1]
    binomialWeightsy[i] = binomial(size[1]-1, i-1);
  end
  # Normilize Vektor
  binomialWeightsy /= sum(binomialWeightsy);

  # Multiplay the two nomilized vektors to get the filter
  f = binomialWeightsy*binomialWeightsx';
  return f::Array{Float64,2}
end

# Downsample an image by a factor of 2
function downsample2(A::Array{Float64,2})
  # Every second row and column is skipped
  D = A[ 1:2:size(A)[1] , 1:2:size(A)[2] ]
  return D::Array{Float64,2}
end

# Upsample an image by a factor of 2
function upsample2(A::Array{Float64,2},fsize::Array{Int,2})

  #   Create empty Upscaled shell
  U = zeros(2*size(A)[1],2*size(A)[2]);
  #   Fill every second pixel with a pixel from A
  U[1:2:size(U)[1],1:2:size(U)[2]] = A;
  #   Create the binomial Filter
  filter = makebinomialfilter(fsize);
  # Use the binomal filter to calclute the missing not defined fields of the upscaled image
  U = imfilter(U,centered(filter), "reflect")
  # Scale every Value by 4
  U = U*4;
  return U::Array{Float64,2}

end

# Build a gaussian pyramid from an image.
# The output array should contain the pyramid levels in decreasing sizes.
function makegaussianpyramid(im::Array{Float32,2},nlevels::Int,fsize::Array{Int,2},sigma::Float64)

  #   Create Gaussian Filter
  gaussianFilter = makegaussianfilter(fsize,sigma);
  #   Create empty shells for the pyramid Images with decreasing size
  G = [zeros( Int(round(size(im)[1]/(2^i),RoundUp)) , Int(round(size(im)[2]/(2^i),RoundUp)) ) for i = 0:nlevels-1]
  #   Save original image in the first layer
  G[1] = im;
  #   Save filtered and downsampled iamges in the appropriate pyramid level
  for i = 2:nlevels
    #   Apply Gaussian Filter
    im = imfilter(im,centered(gaussianFilter),"symmetric");
    #   Downsample Image by two
    im = downsample2(im);
    #   Save modified Image
    G[i] = im;
  end

  return G::Array{Array{Float64,2},1}
end

# Display a given image pyramid (laplacian or gaussian)
function displaypyramid(P::Array{Array{Float64,2},1})

  #   Create empty Figure
  figure()
  #   Width running variable
  width = 1;
  #   Total width of all images stacked next to each other
  totalWidth = 0;
  for i = 1:size(P)[1]
    totalWidth +=size(P[i])[2]
  end
  #   Create Empty image with the size of all images of the pyramid put next to each other
  fig = zeros(size(P[1])[1],totalWidth);
  #   Write all images to their cooresponding position
  for i = 1:size(P)[1]
    fig[ 1:size(P[i])[1] , width:(size(P[i])[2]+width-1) ] = P[i];
    width +=(size(P[i])[2]);
  end
  #   Show the final image
  imshow(fig,"gray");

  return nothing
end

# Build a laplacian pyramid from a gaussian pyramid.
# The output array should contain the pyramid levels in decreasing sizes.
function makelaplacianpyramid(G::Array{Array{Float64,2},1},nlevels::Int,fsize::Array{Int,2})
  # The smalles image stays the same and is saved in the top level of the laplacian Pyramid
  L = G
  #   Calculate and overwrite each Lapacian Image
  for i = 1:nlevels-1
    L[i] = G[i]-upsample2(G[i+1],fsize);
  end
  return L::Array{Array{Float64,2},1}
end

# Amplify frequencies of the first two layers of the laplacian pyramid
function amplifyhighfreq2(L::Array{Array{Float64,2},1})
  #   Amplification factor for the first layer

  amp1 = 2;
  #   Amplification factor for the second layer
  amp2 = 2;

  A = L;
  #   Scale each layer
  A[1] = A[1].*amp1;
  A[2] = A[2].*amp2;

  return A::Array{Array{Float64,2},1}
end

# Reconstruct an image from the laplacian pyramid
function reconstructlaplacianpyramid(L::Array{Array{Float64,2},1},fsize::Array{Int,2})
  # The Top layer again stays the same because it is the Original Top layer
  G = L;
  # Each step down the Pyramid is iteratively calculated from previous levels
  for i = size(L)[1]-1:-1:1
    G[i] = L[i] + upsample2(G[i+1],fsize)
  end
  # The fully reconstructed (bottom layer) image is saved to the output image
  im = G[1];
  return im::Array{Float64,2}
end


# Problem 1: Image Pyramids and Image Sharpening
#
function problem1()
#   # parameters
  fsize = [5 5]
  sigma = 1.4
  nlevels = 6
#
#   # load image
  im = PyPlot.imread("a2p1.png")

#   # create gaussian pyramid
  G = makegaussianpyramid(im,nlevels,fsize,sigma)

#   # display gaussianpyramid
  displaypyramid(G)
  title("Gaussian Pyramid")
#
#   # create laplacian pyramid
  L = makelaplacianpyramid(G,nlevels,fsize)
#
#   # dispaly laplacian pyramid
  displaypyramid(L)
  title("Laplacian Pyramid")
#
#   # amplify finest 2 subands
  L_amp = amplifyhighfreq2(L)
#
#   # reconstruct image from laplacian pyramid
  im_rec = reconstructlaplacianpyramid(L_amp,fsize)
#
#   # display original and reconstructed image
  figure()
  subplot(131)
  imshow(im,"gray",interpolation="none")#,vmin="0",vmax="0.8666667")
  axis("off")
  title("Original Image")
  subplot(132)
  imshow(im_rec,"gray",interpolation="none")#,vmin="0",vmax="0.8666667")
  axis("off")
  title("Reconstructed Image")
  subplot(133)
  imshow(im-im_rec,"gray",interpolation="none")#,vmin="0.05",vmax="1.5")
  axis("off")
  title("Difference")
#   #Display the relusting blurred image
  # imshow(2*im-im_rec,"gray",interpolation="none")
  # axis("off")
  # title("Blurriness")
  gcf()
#
  return
end

problem1()
