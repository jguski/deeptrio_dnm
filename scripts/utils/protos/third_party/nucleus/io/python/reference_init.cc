//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/reference.clif

#include <Python.h>

namespace third__party_nucleus_io_python_reference_clifwrap {

bool Ready();
PyObject* Init();

}  // namespace third__party_nucleus_io_python_reference_clifwrap

PyMODINIT_FUNC PyInit_reference(void) {
  if (!third__party_nucleus_io_python_reference_clifwrap::Ready()) return nullptr;
  return third__party_nucleus_io_python_reference_clifwrap::Init();
}
