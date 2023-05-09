//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/sam_writer.clif

#include <Python.h>
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "clif/python/types.h"
#include "third_party/nucleus/protos/reads_pyclif.h"
#include "third_party/nucleus/util/proto_clif_converter.h"
#include "third_party/nucleus/vendor/statusor_clif_converters.h"
#include "sam_writer.h"
#include "clif/python/stltypes.h"
#include "clif/python/slots.h"

namespace third__party_nucleus_io_python_sam__writer_clifwrap {

using namespace clif;

static const char* ThisModuleName = "third_party.nucleus.io.python.sam_writer";

#define _0 py::postconv::PASS
#define _1 UnicodeFromBytes
#define _2 UnicodeFromBytes

namespace pySamWriter {

struct wrapper {
  PyObject_HEAD
  ::clif::Instance<::nucleus::SamWriter> cpp;
  PyObject* instance_dict = nullptr;
  PyObject* weakrefs = nullptr;
};

static ::nucleus::SamWriter* ThisPtr(PyObject*);

// @classmethod to_file(samPath:str, refPath:str, embedRef:bool, header:SamHeader) -> StatusOr<SamWriter>
static PyObject* wrapToFile_as_to_file(PyObject* cls, PyObject* args, PyObject* kw) {
  PyObject* a[4];
  const char* names[] = {
      "samPath",
      "refPath",
      "embedRef",
      "header",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOOO:to_file", const_cast<char**>(names), &a[0], &a[1], &a[2], &a[3])) return nullptr;
  ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>> arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("to_file", names[0], "::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>", a[0]);
  ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>> arg2;
  if (!Clif_PyObjAs(a[1], &arg2)) return ArgError("to_file", names[1], "::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>", a[1]);
  bool arg3;
  if (!Clif_PyObjAs(a[2], &arg3)) return ArgError("to_file", names[2], "bool", a[2]);
  ::nucleus::genomics::v1::SamHeader arg4;
  if (!Clif_PyObjAs(a[3], &arg4)) return ArgError("to_file", names[3], "::nucleus::genomics::v1::SamHeader", a[3]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::nucleus::StatusOr< ::std::unique_ptr< ::nucleus::SamWriter>> ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::SamWriter::ToFile(std::move(arg1), std::move(arg2), std::move(arg3), std::move(arg4));
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

// write(samMessage:ConstProtoPtr<Read>) -> Status
static PyObject* wrapWritePython_as_write(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "samMessage",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:write", const_cast<char**>(names), &a[0])) return nullptr;
  ::nucleus::ConstProtoPtr<const ::nucleus::genomics::v1::Read> arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("write", names[0], "::nucleus::ConstProtoPtr<const ::nucleus::genomics::v1::Read>", a[0]);
  // Call actual C++ method.
  ::nucleus::SamWriter* c = ThisPtr(self);
  if (!c) return nullptr;
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::tensorflow::Status ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = c->WritePython(std::move(arg1));
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

// __enter__@()
static PyObject* wrapPythonEnter_as___enter__(PyObject* self) {
  // Call actual C++ method.
  ::nucleus::SamWriter* c = ThisPtr(self);
  if (!c) return nullptr;
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    c->PythonEnter();
  } catch(const std::exception& e) {
    err_type = PyExc_RuntimeError;
    err_msg += std::string(": ") + e.what();
  } catch (...) {
    err_type = PyExc_RuntimeError;
  }
  Py_BLOCK_THREADS
  if (err_type) {
    PyErr_SetString(err_type, err_msg.c_str());
    return nullptr;
  }
  Py_INCREF(self);
  return self;
}

// __exit__@() -> Status
static PyObject* wrapClose_as___exit__(PyObject* self) {
  // Call actual C++ method.
  ::nucleus::SamWriter* c = ThisPtr(self);
  if (!c) return nullptr;
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::tensorflow::Status ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = c->Close();
  } catch(const std::exception& e) {
    err_type = PyExc_RuntimeError;
    err_msg += std::string(": ") + e.what();
  } catch (...) {
    err_type = PyExc_RuntimeError;
  }
  Py_BLOCK_THREADS
  if (err_type) {
    PyErr_SetString(err_type, err_msg.c_str());
    return nullptr;
  }
  // Convert return values to Python.
  PyObject* p, * result_tuple = PyTuple_New(1);
  if (result_tuple == nullptr) return nullptr;
  if ((p=Clif_PyObjFrom(std::move(ret0), {})) == nullptr) {
    Py_DECREF(result_tuple);
    return nullptr;
  }
  PyTuple_SET_ITEM(result_tuple, 0, p);
  Py_XDECREF(result_tuple);
  Py_RETURN_NONE;
}

static PyMethodDef MethodsStaticAlloc[] = {
  {"to_file", (PyCFunction)wrapToFile_as_to_file, METH_VARARGS | METH_KEYWORDS | METH_CLASS, "to_file(samPath:str, refPath:str, embedRef:bool, header:SamHeader) -> StatusOr<SamWriter>\n  Calls C++ function\n  ::nucleus::StatusOr< ::std::unique_ptr< ::nucleus::SamWriter>> ::nucleus::SamWriter::ToFile(::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, bool, ::nucleus::genomics::v1::SamHeader)"},
  {"write", (PyCFunction)wrapWritePython_as_write, METH_VARARGS | METH_KEYWORDS, "write(samMessage:ConstProtoPtr<Read>) -> Status\n  Calls C++ function\n  ::tensorflow::Status ::nucleus::SamWriter::WritePython(::nucleus::ConstProtoPtr<const ::nucleus::genomics::v1::Read>)"},
  {"__enter__", (PyCFunction)wrapPythonEnter_as___enter__, METH_NOARGS, "__enter__@()\n  Calls C++ function\n  void ::nucleus::SamWriter::PythonEnter()"},
  {"__exit__", (PyCFunction)wrapClose_as___exit__, METH_VARARGS | METH_KEYWORDS, "__exit__@() -> Status\n  Calls C++ function\n  ::tensorflow::Status ::nucleus::SamWriter::Close()"},
  {"__reduce_ex__", (PyCFunction)::clif::ReduceExImpl, METH_VARARGS | METH_KEYWORDS, "Helper for pickle."},
  {}
};

// SamWriter __new__
static PyObject* _new(PyTypeObject* type, Py_ssize_t nitems);

// SamWriter __del__
static void _dtor(PyObject* self) {
  if (reinterpret_cast<wrapper*>(self)->weakrefs) {
    PyObject_ClearWeakRefs(self);
  }
  Py_BEGIN_ALLOW_THREADS
  reinterpret_cast<wrapper*>(self)->cpp.Destruct();
  Py_END_ALLOW_THREADS
  Py_TYPE(self)->tp_free(self);
}

static void _del(void* self) {
  delete reinterpret_cast<wrapper*>(self);
}

PyTypeObject* wrapper_Type = nullptr;

static PyTypeObject* _build_heap_type() {
  PyHeapTypeObject *heap_type =
      (PyHeapTypeObject *) PyType_Type.tp_alloc(&PyType_Type, 0);
  if (!heap_type)
    return nullptr;
  heap_type->ht_qualname = (PyObject *) PyUnicode_FromString(
      "SamWriter");
  Py_INCREF(heap_type->ht_qualname);
  heap_type->ht_name = heap_type->ht_qualname;
  PyTypeObject *ty = &heap_type->ht_type;
  ty->tp_as_number = &heap_type->as_number;
  ty->tp_as_sequence = &heap_type->as_sequence;
  ty->tp_as_mapping = &heap_type->as_mapping;
#if PY_VERSION_HEX >= 0x03050000
  ty->tp_as_async = &heap_type->as_async;
#endif
  ty->tp_name = "third_party.nucleus.io.python.sam_writer.SamWriter";
  ty->tp_basicsize = sizeof(wrapper);
  ty->tp_dealloc = _dtor;
  ty->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
  ty->tp_doc = "CLIF wrapper for ::nucleus::SamWriter";
  ty->tp_methods = MethodsStaticAlloc;
  ty->tp_init = Clif_PyType_Inconstructible;
  ty->tp_alloc = _new;
  ty->tp_new = PyType_GenericNew;
  ty->tp_free = _del;
  ty->tp_weaklistoffset = offsetof(wrapper, weakrefs);
  return ty;
}

static PyObject* _new(PyTypeObject* type, Py_ssize_t nitems) {
  DCHECK(nitems == 0);
  wrapper* wobj = new wrapper;
  PyObject* self = reinterpret_cast<PyObject*>(wobj);
  return PyObject_Init(self, wrapper_Type);
}

static ::nucleus::SamWriter* ThisPtr(PyObject* py) {
  if (Py_TYPE(py) == wrapper_Type) {
    return ::clif::python::Get(reinterpret_cast<wrapper*>(py)->cpp);
  }
  PyObject* base = PyObject_CallMethod(py, "as_nucleus_SamWriter", nullptr);
  if (base == nullptr) {
    PyErr_Clear();
  } else {
    if (PyCapsule_CheckExact(base)) {
      void* p = PyCapsule_GetPointer(base, "::nucleus::SamWriter");
      if (!PyErr_Occurred()) {
        ::nucleus::SamWriter* c = static_cast<::nucleus::SamWriter*>(p);
        Py_DECREF(base);
        return c;
      }
    }
    Py_DECREF(base);
  }
  if (PyObject_IsInstance(py, reinterpret_cast<PyObject*>(wrapper_Type))) {
    if (!base) {
      return ::clif::python::Get(reinterpret_cast<wrapper*>(py)->cpp);
    }
    PyErr_Format(PyExc_ValueError, "can't convert %s %s to ::nucleus::SamWriter*", ClassName(py), ClassType(py));
  } else {
    PyErr_Format(PyExc_TypeError, "expecting %s instance, got %s %s", wrapper_Type->tp_name, ClassName(py), ClassType(py));
  }
  return nullptr;
}

}  // namespace pySamWriter

// Initialize module

bool Ready() {
  pySamWriter::wrapper_Type =
  pySamWriter::_build_heap_type();
  if (PyType_Ready(pySamWriter::wrapper_Type) < 0) return false;
  PyObject *modname = PyUnicode_FromString(ThisModuleName);
  if (modname == nullptr) return false;
  PyObject_SetAttrString((PyObject *) pySamWriter::wrapper_Type, "__module__", modname);
  Py_INCREF(pySamWriter::wrapper_Type);  // For PyModule_AddObject to steal.
  return true;
}

static struct PyModuleDef Module = {
  PyModuleDef_HEAD_INIT,
  ThisModuleName,
  "CLIF-generated module for third_party/nucleus/io/python/sam_writer.clif", // module doc
  -1,  // module keeps state in global variables
  nullptr,
  nullptr,  // m_slots a.k.a. m_reload
  nullptr,  // m_traverse
  ClearImportCache  // m_clear
};

PyObject* Init() {
  PyObject* module = PyModule_Create(&Module);
  if (!module) return nullptr;
  PyEval_InitThreads();
  if (PyModule_AddObject(module, "SamWriter", reinterpret_cast<PyObject*>(pySamWriter::wrapper_Type)) < 0) goto err;
  return module;
err:
  Py_DECREF(module);
  return nullptr;
}

}  // namespace third__party_nucleus_io_python_sam__writer_clifwrap

namespace nucleus {
using namespace ::clif;
using ::clif::Clif_PyObjAs;
using ::clif::Clif_PyObjFrom;

// SamWriter to/from ::nucleus::SamWriter conversion

bool Clif_PyObjAs(PyObject* py, ::nucleus::SamWriter** c) {
  CHECK(c != nullptr);
  if (Py_None == py) {
    *c = nullptr;
    return true;
  }
  ::nucleus::SamWriter* cpp = third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = cpp;
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::shared_ptr<::nucleus::SamWriter>* c) {
  CHECK(c != nullptr);
  ::nucleus::SamWriter* cpp = third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = ::clif::MakeStdShared(reinterpret_cast<third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::wrapper*>(py)->cpp, cpp);
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::unique_ptr<::nucleus::SamWriter>* c) {
  CHECK(c != nullptr);
  ::nucleus::SamWriter* cpp = third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::ThisPtr(py);
  if (cpp == nullptr) return false;
  if (!reinterpret_cast<third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::wrapper*>(py)->cpp.Detach()) {
    PyErr_SetString(PyExc_ValueError, "Cannot convert SamWriter instance to std::unique_ptr.");
    return false;
  }
  c->reset(cpp);
  return true;
}

PyObject* Clif_PyObjFrom(::nucleus::SamWriter* c, py::PostConv unused) {
  CHECK(third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::wrapper_Type != nullptr) <<
    "---> Function Clif_PyObjFrom(::nucleus::SamWriter) called before " <<
    third__party_nucleus_io_python_sam__writer_clifwrap::ThisModuleName  <<
    " was imported from Python.";
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::wrapper_Type, NULL, NULL);
  reinterpret_cast<third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::wrapper*>(py)->cpp = ::clif::Instance<::nucleus::SamWriter>(c, ::clif::UnOwnedResource());
  return py;
}

PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::SamWriter> c, py::PostConv unused) {
  CHECK(third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::wrapper_Type != nullptr) <<
    "---> Function Clif_PyObjFrom(::nucleus::SamWriter) called before " <<
    third__party_nucleus_io_python_sam__writer_clifwrap::ThisModuleName  <<
    " was imported from Python.";
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::wrapper_Type, NULL, NULL);
  reinterpret_cast<third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::wrapper*>(py)->cpp = ::clif::Instance<::nucleus::SamWriter>(c);
  return py;
}

PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::SamWriter> c, py::PostConv unused) {
  CHECK(third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::wrapper_Type != nullptr) <<
    "---> Function Clif_PyObjFrom(::nucleus::SamWriter) called before " <<
    third__party_nucleus_io_python_sam__writer_clifwrap::ThisModuleName  <<
    " was imported from Python.";
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::wrapper_Type, NULL, NULL);
  reinterpret_cast<third__party_nucleus_io_python_sam__writer_clifwrap::pySamWriter::wrapper*>(py)->cpp = ::clif::Instance<::nucleus::SamWriter>(std::move(c));
  return py;
}

}  // namespace nucleus