//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/fastq_reader.clif

#include <Python.h>

namespace third__party_nucleus_io_python_fastq__reader_clifwrap {

bool Ready();
PyObject* Init();

}  // namespace third__party_nucleus_io_python_fastq__reader_clifwrap

PyMODINIT_FUNC PyInit_fastq_reader(void) {
  if (!third__party_nucleus_io_python_fastq__reader_clifwrap::Ready()) return nullptr;
  return third__party_nucleus_io_python_fastq__reader_clifwrap::Init();
}
