//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/bedgraph_writer.clif

#include <Python.h>

namespace third__party_nucleus_io_python_bedgraph__writer_clifwrap {

bool Ready();
PyObject* Init();

}  // namespace third__party_nucleus_io_python_bedgraph__writer_clifwrap

PyMODINIT_FUNC PyInit_bedgraph_writer(void) {
  if (!third__party_nucleus_io_python_bedgraph__writer_clifwrap::Ready()) return nullptr;
  return third__party_nucleus_io_python_bedgraph__writer_clifwrap::Init();
}
