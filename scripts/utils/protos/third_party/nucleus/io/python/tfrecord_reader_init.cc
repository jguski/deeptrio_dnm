//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/tfrecord_reader.clif

#include <Python.h>

namespace third__party_nucleus_io_python_tfrecord__reader_clifwrap {

bool Ready();
PyObject* Init();

}  // namespace third__party_nucleus_io_python_tfrecord__reader_clifwrap

PyMODINIT_FUNC PyInit_tfrecord_reader(void) {
  if (!third__party_nucleus_io_python_tfrecord__reader_clifwrap::Ready()) return nullptr;
  return third__party_nucleus_io_python_tfrecord__reader_clifwrap::Init();
}
