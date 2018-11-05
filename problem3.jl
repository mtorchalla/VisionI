using Images
using PyPlot
using Test
using LinearAlgebra
using FileIO

# Transform from Cartesian to homogeneous coordinates
function cart2hom(points::Array{Float64,2})
  points_hom = vcat(points, ones(1,size(points)[2]))
  return points_hom::Array{Float64,2}
end


# Transform from homogeneous to Cartesian coordinates
function hom2cart(points::Array{Float64,2})
  points_cart = points[1:end-1,:]
  return points_cart::Array{Float64,2}
end


# Translation by v
function gettranslation(v::Array{Float64,1})
  T = hcat([1 0 0; 0 1 0; 0 0 1], v)
  return T::Array{Float64,2}
end

# Rotation of d degrees around x axis
function getxrotation(d::Int)
  Rx = [1 0 0; 0 cosd(d) -sind(d); 0 sind(d) cosd(d)]
  return Rx::Array{Float64,2}
end

# Rotation of d degrees around y axis
function getyrotation(d::Int)
  Ry = [cosd(d) 0 sind(d); 0 1 0; -sind(d) 0 cosd(d)]
  return Ry::Array{Float64,2}
end

# Rotation of d degrees around z axis
function getzrotation(d::Int)
  Rz = [cosd(d) -sind(d) 0; sind(d) cosd(d) 0; 0 0 1]
  return Rz::Array{Float64,2}
end


# Central projection matrix (including camera intrinsics)
function getcentralprojection(principal::Array{Int,1}, focal::Int)
  K = [focal 0 principal[1]; 0 focal principal[2]; 0 0 1] * 1.0 #Implicit typecast to Float64
  return K::Array{Float64,2}
end


# Return full projection matrix P and full model transformation matrix M
function getfullprojection(T::Array{Float64,2},Rx::Array{Float64,2},Ry::Array{Float64,2},Rz::Array{Float64,2},V::Array{Float64,2})
  #The order of the rotation transformation is in general not kommutativ,
  #because the multiplication of matrices is not kommutativ, too
  #The translation however is kommutativ, because we can add the vectors together
  P =  V*hcat(Ry*Rx*Rz, T[:,4])
  M = vcat(hcat(Ry*Rx*Rz, T[:,4]), [0 0 0 1])
  return P::Array{Float64,2},M::Array{Float64,2}
end


# Load 2D points
function loadpoints()
  points = load("obj2d.jld2", "x")
  return points::Array{Float64,2}
end


# Load z-coordinates
function loadz()
  z = load("zs.jld2", "Z")
  return z::Array{Float64,2}
end


# Invert just the central projection P of 2d points *P2d* with z-coordinates *z*
function invertprojection(P::Array{Float64,2}, P2d::Array{Float64,2}, z::Array{Float64,2})
  P3d = inv(P) * ( cart2hom(P2d) .* z )
  return P3d::Array{Float64,2}
end


# Invert just the model transformation of the 3D points *P3d*
function inverttransformation(A::Array{Float64,2}, P3d::Array{Float64,2})
  X = inv(A) * cart2hom(P3d)
  return X::Array{Float64,2}
end


# Plot 2D points
function displaypoints2d(points::Array{Float64,2})
  figure()
  plot(points[1,:], points[2,:])
  title("2D-Points")
  return gcf()::Figure
end

# Plot 3D points
function displaypoints3d(points::Array{Float64,2})
  figure()
  plot3D(points[1,:], points[2,:], points[3,:]) #more performant than scatter3D.
  title("It's a cow! (3D-Points)")
  return gcf()::Figure
end

# Apply full projection matrix *C* to 3D points *X*
function projectpoints(P::Array{Float64,2}, X::Array{Float64,2})
  P2d = P * cart2hom(X)
  P2d = P2d./P2d[3,:]' # Divide the camera coordinates by Z to get the image coordinates
  P2d = hom2cart(P2d)
  return P2d::Array{Float64,2}
end



#= Problem 2
Projective Transformation =#

function problem3()
  # parameters
  t               = [6.7; -10; 4.2]
  principal_point = [9; -7]
  focal_length    = 8

  # model transformations
  T = gettranslation(t)
  Ry = getyrotation(-45)
  Rx = getxrotation(120)
  Rz = getzrotation(-10)

  # central projection including camera intrinsics
  K = getcentralprojection(principal_point,focal_length)

  # full projection and model matrix
  P,M = getfullprojection(T,Rx,Ry,Rz,K)

  # load data and plot it
  points = loadpoints()
  displaypoints2d(points)

  # reconstruct 3d scene
  z = loadz()
  #Its necessary to provide the z-coordinates of every Points, because this
  #Information gets lost while projecting the 3D-Environment to the 2D Camera Image
  Xt = invertprojection(K,points,z)
  Xh = inverttransformation(M,Xt)

  worldpoints = hom2cart(Xh)
  displaypoints3d(worldpoints)

  # reproject points
  points2 = projectpoints(P,worldpoints)

  displaypoints2d(points2)

  @test points ≈ points2
  return
end

problem3()
