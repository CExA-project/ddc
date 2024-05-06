# Transformations {#transformations}
<!--
Copyright (C) The DDC development team, see COPYRIGHT.md file

SPDX-License-Identifier: MIT
-->

On the top of DDC core - which provides the basic tools for the needs of performance-portable physical simulations
(describing meshes with *DiscreteDomain* and representing physical fields with *Chunk*) - are build computation kernels
to perform transformations. At the moment, the two available transformations are Discrete Fourier Transform and Spline methods.

Those methods are performance portable (supported with CPU Serial, CPU OpenMP, CUDA and HIP backends). They are used
in *examples/heat_equation_spectral.cpp* (for DFT) and *examples/characteristics_advection.cpp* (for splines).
A particular use-case of splines is benchmarked in *benchmarks/splines.cpp*.

## General presentation of transformations

### Computation kernels dedicated to transformations in DDC

A transformation is a change of basis, going from a function space to another. Because the available memory is finite,
the bases are necessarily finite (which means it contains a finite set of basis functions, but those basis functions
may be continuous).

There are many use-cases of transformations for simulation or signal processing needs: interpolation, spectral methods,
finite element methods, filtering (compression), signal analysis (post-process)...

Every basis has its own specificities (ie. orthogonality, n-derivability, etc...) but as they belong to the same frame
of "set of functions which form a basis", they have in common a formalism and a terminology. However, this is currently
not very manifest when looking at the API in the DDC implementations (DFT and Spline API are very different).
It must be explained:

- Fourier requires the periodicity of the represented function, thus boundary conditions does not need to be provided.
- Fourier basis functions are indexed with the wave vector k, whose possible values form a set of coordinates which generates
the Fourier space. Mesh of "real" space and mesh of Fourier space are in bijection one-to-the-other.
The situation is more complicated for Splines.
- Fourier and Splines basis use-cases are quite different. Thus - as shown below - the so-called "evaluator" is not
implemented for Fourier (while inverse DFT is) and inverse Spline transform is not implemented (while Spline evaluation is).
- FFT backends support 1D to 3D Fourier transforms. General n-dimensional Spline transform would be a mess to implement for all supported boundary conditions so we limit to the 2D case.
- DFT is not currently supported in batched configuration (but it may be the subject of future development). Batched splines are supported.

To summarize the available features of the two kernels:

|          | Direct transformation | Inverse transformation | Evaluation |
|----------|-----------------|--------|-------------------|
| DFT      | *fft*           | *ifft* | x                 |
| Spline   | *SplineBuilder* | x      | *SplineEvaluator* |

So, even if the two kernels *could* share a common API, this is not currently the case but this is more due to pratical considerations than intrinsic mathematical differences between the two methods.

### General transformations theory

The simplest way to describe a discrete function \f$ f \f$ is to provide its value \f$ f(x_i) \f$ at every point of the mesh which supports it.

Another more general method is to define a basis \f$ \phi_j \f$ in a function space and provide the set of coefficients
\f$ c_j \f$ which generate the discrete function. The set of coefficients is thus another way to represent the discrete
function:

\f[
f(x_i) = \sum_j c_j \phi_j(x_i)
\f]

Note: the first method is covered by the second one in the particular case of \f$ \phi \f$ being the Dirac basis functions.

The aim of a computation kernel dedicated to a transformation is to determine the values of coefficients \f$ c_j \f$
knowing \f$ f(x_i) \f$ (and for the inverse transformation, \f$ f(x_i) \f$ knowing \f$ c_j \f$). 

Additionally to the *transformation* and the *inverse transformation*, there is a third operation which can be useful:
the *evaluation*. Indeed, more than a discrete function, coefficients \f$ c_j \f$ generate an interpolating (continuous)
function which can be evaluated anywhere between the points of the mesh.

Note: performing a (direct) transformation then an evaluation is an interpolation.

Bases of function spaces can have additional properties like orthogonality
\f$ <\phi_i|\phi_j>= \int \phi_i(x) \phi_j(x)\; dx = \delta_{ij} \f$ (thus, \f$ c_j = <f|\phi_j> \f$).
This is the case for Fourier transform, and it provides a numerical method to perform the transformation (but not the most
efficient one). In the general case orthogonality is not verified though, thus transformation algorithms are basis-specific.

### Lexical 

Here is a summary of different general terms used in the context of DDC transformations:

| Term     | Meaning & commentary |
|----------|-----------------|
| Discrete function | Function defined on a mesh (via its values at each point of via its coefficients in a particular finite basis representation). |
| Dimension(s) of interest | Dimension(s) along which the transformation is performed |
| Batch dimension(s) | Dimension(s) unaffected by the transformation. It leads to a set of transformations performed independantly (along dimensions of interest). Those are embarassingly parallelizable. |
| Interpolating function | Continuous function generated from the coefficients \f$ c_j \f$ and the basis functions \f$ \phi_j \f$. |
| Approximation | Same as transformation (computing the coefficients of the interpolating function). |
| Interpolation points | Points of the mesh, supporting the discrete function through which interpolating function goes. |
| Evaluation points | Points on which we want the interpolating function to be evaluated. |

## Discrete Fourier Transform

Fourier transform is the transformation with \f$ \phi_j=e^{ikx} \f$. The algorithm used is the Fast Fourier Transform.

Fourier space associated to continuous dimension *X* is represented using the *Fourier<X>* templated tag.

The performance-compatibility is ensured using the backends FFTW, cuFFT or hipFFT depending of Kokkos::ExecutionSpace
which templates the functions. Thus, no FFT algorithm is actually implemented in DDC, which is more a wrapper of those
backends. However, the spectral mesh can be computed (as a *DiscreteDomain*) knowing the real mesh (which is not a
commonly-implemented feature in the well-known FFT libraries).

## Spline transform

Please refer to Emily Bourne's thesis (https://www.theses.fr/2022AIXM0412.pdf) and the spline-related API documentation.
