//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/tabix_indexer.clif

#include <Python.h>
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "clif/python/types.h"
#include "third_party/nucleus/vendor/statusor_clif_converters.h"
#include "tabix_indexer.h"
#include "clif/python/stltypes.h"
#include "clif/python/slots.h"

namespace third__party_nucleus_io_python_tabix__indexer_clifwrap {

using namespace clif;

static const char* ThisModuleName = "third_party.nucleus.io.python.tabix_indexer";

#define _0 py::postconv::PASS
#define _1 UnicodeFromBytes
#define _2 UnicodeFromBytes

// tbx_index_build(path:str) -> Status
static PyObject* wrapTbxIndexBuild_as_tbx_index_build(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "path",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:tbx_index_build", const_cast<char**>(names), &a[0])) return nullptr;
  ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>> arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("tbx_index_build", names[0], "::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>", a[0]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::tensorflow::Status ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::TbxIndexBuild(std::move(arg1));
  } catch(const std::exception& e) {
    err_type = PyExc_RuntimeError;
    err_msg += std::string(": ") + e.what();
  } catch (...) {
    err_type = PyExc_RuntimeError;
  }
  Py_BLOCK_THREADS
  Py_DECREF(args);
  Py_XDECREF(kw);
  if (err_type) {
    PyErr_SetString(err_type, err_msg.c_str());
    return nullptr;
  }
  return Clif_PyObjFrom(std::move(ret0), {});
}

// csi_index_build(path:str, min_shift:int) -> Status
static PyObject* wrapCSIIndexBuild_as_csi_index_build(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[2];
  const char* names[] = {
      "path",
      "min_shift",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO:csi_index_build", const_cast<char**>(names), &a[0], &a[1])) return nullptr;
  ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>> arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("csi_index_build", names[0], "::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>", a[0]);
  int arg2;
  if (!Clif_PyObjAs(a[1], &arg2)) return ArgError("csi_index_build", names[1], "int", a[1]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::tensorflow::Status ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::CSIIndexBuild(std::move(arg1), std::move(arg2));
  } catch(const std::exception& e) {
    err_type = PyExc_RuntimeError;
    err_msg += std::string(": ") + e.what();
  } catch (...) {
    err_type = PyExc_RuntimeError;
  }
  Py_BLOCK_THREADS
  Py_DECREF(args);
  Py_XDECREF(kw);
  if (err_type) {
    PyErr_SetString(err_type, err_msg.c_str());
    return nullptr;
  }
  return Clif_PyObjFrom(std::move(ret0), {});
}

// Initialize module

static PyMethodDef MethodsStaticAlloc[] = {
  {"tbx_index_build", (PyCFunction)wrapTbxIndexBuild_as_tbx_index_build, METH_VARARGS | METH_KEYWORDS, "tbx_index_build(path:str) -> Status\n  Calls C++ function\n  ::tensorflow::Status ::nucleus::TbxIndexBuild(::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>)"},
  {"csi_index_build", (PyCFunction)wrapCSIIndexBuild_as_csi_index_build, METH_VARARGS | METH_KEYWORDS, "csi_index_build(path:str, min_shift:int) -> Status\n  Calls C++ function\n  ::tensorflow::Status ::nucleus::CSIIndexBuild(::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, int)"},
  {}
};

bool Ready() {
  return true;
}

static struct PyModuleDef Module = {
  PyModuleDef_HEAD_INIT,
  ThisModuleName,
  "CLIF-generated module for third_party/nucleus/io/python/tabix_indexer.clif", // module doc
  -1,  // module keeps state in global variables
  MethodsStaticAlloc,
  nullptr,  // m_slots a.k.a. m_reload
  nullptr,  // m_traverse
  ClearImportCache  // m_clear
};

PyObject* Init() {
  PyObject* module = PyModule_Create(&Module);
  if (!module) return nullptr;
  PyEval_InitThreads();
  return module;
}

}  // namespace third__party_nucleus_io_python_tabix__indexer_clifwrap
