//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/vcf_reader.clif

#include <Python.h>

namespace third__party_nucleus_io_python_vcf__reader_clifwrap {

bool Ready();
PyObject* Init();

}  // namespace third__party_nucleus_io_python_vcf__reader_clifwrap

PyMODINIT_FUNC PyInit_vcf_reader(void) {
  if (!third__party_nucleus_io_python_vcf__reader_clifwrap::Ready()) return nullptr;
  return third__party_nucleus_io_python_vcf__reader_clifwrap::Init();
}
