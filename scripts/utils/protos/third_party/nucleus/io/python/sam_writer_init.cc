//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/sam_writer.clif

#include <Python.h>

namespace third__party_nucleus_io_python_sam__writer_clifwrap {

bool Ready();
PyObject* Init();

}  // namespace third__party_nucleus_io_python_sam__writer_clifwrap

PyMODINIT_FUNC PyInit_sam_writer(void) {
  if (!third__party_nucleus_io_python_sam__writer_clifwrap::Ready()) return nullptr;
  return third__party_nucleus_io_python_sam__writer_clifwrap::Init();
}
