\page first_steps First steps: the heat equation


# Definition of a mesh

\snippet first_steps/heat_equation.cpp mesh

# Domain of interest

  We initialize the full and the inner domain using DiscreteVectors that contain information on the spread of points in our space. For the inner domain, we further precise its origin using a DiscreteCoordinate as it is not the same as our space's origin.

  Ghost points will be used for stencil computation on inner domain boundaries' points.
  ![domains_image](./images/domains.png "Domains")

\snippet first_steps/heat_equation.cpp domain

# Memory allocation
  We allocate two 2D Chunks along the X and Y dimensions which will be used to map temperature to the domains' points at t and t+dt.

\snippet first_steps/heat_equation.cpp memory allocation

# Subdomains
  We define ChunkSpans corresponding to the inner and ghost borders. These will be used for the periodic boundary conditions.
\snippet first_steps/heat_equation.cpp subdomains

# Numerical scheme
  Computes a single time step and stores the result in T_out. Iterating this step a chosen number of times while updating the boundary conditions and the chunks is our end goal.
\snippet first_steps/heat_equation.cpp numerical scheme

# IO

\snippet first_steps/heat_equation.cpp io/pdi
