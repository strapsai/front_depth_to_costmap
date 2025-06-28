#include <pybind11/pybind11.h>
#include <pybind11/numpy.h> 

#include "nav2_costmap_2d/nav2_costmap_2d/costmap_2d.hpp"

namespace py = pybind11;
namespace nav2 = nav2_costmap_2d;

PYBIND11_MODULE(costmap_2d_py, m)
{
    m.doc() = "This is a Python binding of nav2_costmap_2d for real-time updates";

    py::class_<nav2::Costmap2D>(m, "Costmap2D")
        .def(py::init<unsigned int, unsigned int, double,
                     double, double, unsigned char>())
        .def("setCost", &nav2::Costmap2D::setCost, "Set the cost of a cell") // <<< 필수 기능
        .def("getCost", static_cast<unsigned char (nav2::Costmap2D::*)(unsigned int, unsigned int) const>(&nav2::Costmap2D::getCost), "Get the cost of a cell")
        .def("updateOrigin", &nav2::Costmap2D::updateOrigin, "Update the origin of the costmap") // <<< 필수 기능
        .def("resetMap",
             static_cast<void (nav2::Costmap2D::*)(unsigned int, unsigned int, unsigned int, unsigned int)>(&nav2::Costmap2D::resetMap),
             "Resets the costs of a rectangle in the costmap") // <<< 필수 기능
        .def("getSizeInCellsX", &nav2::Costmap2D::getSizeInCellsX, "Get the size of the map in cells (x-axis)") // <<< 필수 기능
        .def("getSizeInCellsY", &nav2::Costmap2D::getSizeInCellsY, "Get the size of the map in cells (y-axis)") // <<< 필수 기능
        .def("getResolution", &nav2::Costmap2D::getResolution, "Get the resolution of the map") // <<< 필수 기능
        .def("getOriginX", &nav2::Costmap2D::getOriginX, "Get the origin x of the map") // <<< 필수 기능
        .def("getOriginY", &nav2::Costmap2D::getOriginY, "Get the origin y of the map") // <<< 필수 기능
        .def("getCharMap", [](nav2::Costmap2D &self) {
            unsigned char* char_map = self.getCharMap();
            unsigned int size_x = self.getSizeInCellsX();
            unsigned int size_y = self.getSizeInCellsY();
            return py::array_t<unsigned char>({size_y, size_x}, {sizeof(unsigned char) * size_x, sizeof(unsigned char)}, char_map);
        }, "Get the costmap data as a NumPy array"); // <<< 필수 기능
}