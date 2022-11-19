#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cudecomp.h>
#include <mpi.h>
#include "checks.h"
#include "helpers.h"
#include "jaxdecomp.h"

namespace py = pybind11;
namespace jd = jaxdecomp;

namespace jaxdecomp {

    // Global cuDecomp handle, initialized once from Python when the
    // library is imported, and then implicitly reused in all functions
    cudecompHandle_t handle;

    /**
    * @brief Initializes the global handle
    */
    void init(){
        CHECK_CUDECOMP_EXIT(cudecompInit(&handle, MPI_COMM_WORLD));
    };
    
    /**
    * @brief Finalizes the cuDecomp library
    */
    void finalize(){
        CHECK_CUDECOMP_EXIT(cudecompFinalize(handle));
    };

    /**
    * @brief Returns Pencil information for a given grid
    */
    decompPencilInfo_t getPencilInfo(decompGridDescConfig_t grid_config){ 
        // Create cuDecomp grid descriptor
        cudecompGridDescConfig_t config;
        cudecompGridDescConfigSet(&config, &grid_config);
        // Create the grid description
        cudecompGridDesc_t grid_desc;
        CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));
        cudecompPencilInfo_t pencil_info;
        CHECK_CUDECOMP_EXIT(cudecompGetPencilInfo(handle, grid_desc, &pencil_info, 2, nullptr));
        decompPencilInfo_t result;
        decompPencilInfoSet(&result, &pencil_info);
        return result;
    };

    /// XLA interface ops

    /** 
    * @brief Wrapper to cudecompTransposeXToY 
    * Transpose data from X-axis aligned pencils to a Y-axis aligned pencils.
    */
    void transposeXtoY(cudaStream_t stream, void** buffers,
                       const char* opaque, size_t opaque_len){

        void* data_d = buffers[0]; // In place operations, so only one buffer
 
        // Create cuDecomp grid descriptor
        cudecompGridDescConfig_t config;
        cudecompGridDescConfigSet(&config, UnpackDescriptor<decompGridDescConfig_t>(opaque, opaque_len));
        
        // Create the grid description
        cudecompGridDesc_t grid_desc;
        CHECK_CUDECOMP_EXIT(cudecompGridDescCreate(handle, &grid_desc, &config, nullptr));

        // Get workspace sizes
        int64_t transpose_work_num_elements;
        CHECK_CUDECOMP_EXIT(cudecompGetTransposeWorkspaceSize(handle, grid_desc, &transpose_work_num_elements));

        int64_t dtype_size;
        CHECK_CUDECOMP_EXIT(cudecompGetDataTypeSize(CUDECOMP_FLOAT, &dtype_size));

        double* transpose_work_d;
        CHECK_CUDECOMP_EXIT(cudecompMalloc(handle, grid_desc, reinterpret_cast<void**>(&transpose_work_d),
                                            transpose_work_num_elements * dtype_size));

        CHECK_CUDECOMP_EXIT(
            cudecompTransposeXToY(handle, grid_desc, data_d, data_d, transpose_work_d, CUDECOMP_FLOAT, nullptr, nullptr, 0));

        CHECK_CUDECOMP_EXIT(cudecompFree(handle, grid_desc, transpose_work_d));

        CHECK_CUDECOMP_EXIT(cudecompGridDescDestroy(handle, grid_desc));
    }

    // Utility to export ops to XLA
    py::dict Registrations() {
        py::dict dict;
        dict["transpose_x_y"] = EncapsulateFunction(transposeXtoY);
        return dict;
    }
}

PYBIND11_MODULE(_jaxdecomp, m) {
    // Utilities
    m.def("init", &jd::init);
    m.def("finalize", &jd::finalize);
    m.def("get_pencil_info", &jd::getPencilInfo);

    // Function registering the custom ops
    m.def("registrations", &jd::Registrations);

    // Utilities for exported ops
    m.def("build_grid_config_descriptor", 
    []( jd::decompGridDescConfig_t config) { return PackDescriptor(config); }); 

    // Exported types
    py::enum_<cudecompTransposeCommBackend_t>(m, "TransposeCommBackend")
    .value("TRANSPOSE_COMM_MPI_P2P", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_MPI_P2P)
    .value("TRANSPOSE_COMM_MPI_P2P_PL", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_MPI_P2P_PL)
    .value("TRANSPOSE_COMM_MPI_A2A", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_MPI_A2A)
    .value("TRANSPOSE_COMM_NCCL", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_NCCL)
    .value("TRANSPOSE_COMM_NCCL_PL", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_NCCL_PL)
    .value("TRANSPOSE_COMM_NVSHMEM", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_NVSHMEM)
    .value("TRANSPOSE_COMM_NVSHMEM_PL", cudecompTransposeCommBackend_t::CUDECOMP_TRANSPOSE_COMM_NVSHMEM_PL)
    .export_values();

    py::enum_<cudecompHaloCommBackend_t>(m, "HaloCommBackend")
    .value("HALO_COMM_MPI", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_MPI)
    .value("HALO_COMM_MPI_BLOCKING", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_MPI_BLOCKING)
    .value("HALO_COMM_NCCL", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_NCCL)
    .value("HALO_COMM_NVSHMEM", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_NVSHMEM)
    .value("HALO_COMM_NVSHMEM_BLOCKING", cudecompHaloCommBackend_t::CUDECOMP_HALO_COMM_NVSHMEM_BLOCKING)
    .export_values();

    py::class_<jd::decompPencilInfo_t> pencil_info(m, "PencilInfo");
    pencil_info.def(py::init<>())
                .def_readonly("shape", &jd::decompPencilInfo_t::shape)
                .def_readonly("lo", &jd::decompPencilInfo_t::lo)
                .def_readonly("hi", &jd::decompPencilInfo_t::hi)
                .def_readonly("order", &jd::decompPencilInfo_t::order)
                .def_readonly("halo_extents", &jd::decompPencilInfo_t::halo_extents)
                .def_readonly("size", &jd::decompPencilInfo_t::size);

    py::class_<jaxdecomp::decompGridDescConfig_t> config(m, "GridConfig");
    config.def(py::init<>())
        .def_readwrite("gdims", &jd::decompGridDescConfig_t::gdims)
        .def_readwrite("pdims", &jd::decompGridDescConfig_t::pdims)
        .def_readwrite("transpose_comm_backend", &jd::decompGridDescConfig_t::transpose_comm_backend)
        .def_readwrite("halo_comm_backend", &jd::decompGridDescConfig_t::halo_comm_backend);

}
