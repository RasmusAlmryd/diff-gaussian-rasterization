/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

typedef struct{
  int a;
  int b;
} example;

struct Test{
  int k;
  int t;

  Test(int k, int t): k(k), t(k){}
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("adamUpdate", &adamUpdate);
  m.def("gaussNewtonUpdate", &gaussNewtonUpdate);
  m.def("gaussNewtonUpdateSimple", &gaussNewtonUpdateSimple);
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);
  // py::class_<example>(m, "example")
  //       .def(py::init<>());
  // py::class_<Test>(m, "Test")
  //       .def(py::init<int, int>())
  //       .def_readwrite("k", &Test::k)
  //       .def_readwrite("t", &Test::t);
  py::class_<Raster_settings>(m, "Raster_settings")
        .def(py::init<int, int, float, float,
          torch::Tensor, float, torch::Tensor, torch::Tensor, int,
          torch::Tensor, bool, bool, bool, int, int>());
  // //       .def_readwrite("image_height", &Raster_settings::image_height)
  // //       .def_readwrite("image_width", &Raster_settings::image_width)
  // //       .def_readwrite("tanfovx", &Raster_settings::tanfovx)
  // //       .def_readwrite("tanfovy", &Raster_settings::tanfovy)
  // //       .def_readwrite("bg", &Raster_settings::bg)
  // //       .def_readwrite("scale_modifier", &Raster_settings::scale_modifier)
  // //       .def_readwrite("viewmatrix", &Raster_settings::viewmatrix)
  // //       .def_readwrite("projmatrix", &Raster_settings::projmatrix)
  // //       .def_readwrite("sh_degree", &Raster_settings::sh_degree)
  // //       .def_readwrite("campos", &Raster_settings::campos)
  // //       .def_readwrite("prefiltered", &Raster_settings::prefiltered)
  // //       .def_readwrite("debug", &Raster_settings::debug)
  // //       .def_readwrite("antialiasing", &Raster_settings::antialiasing)
  // //       .def_readwrite("num_views", &Raster_settings::num_views)
  // //       .def_readwrite("view_index", &Raster_settings::view_index);
}

