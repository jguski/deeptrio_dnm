//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/tfrecord_writer.clif

#include <Python.h>
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "clif/python/types.h"
#include "tfrecord_writer.h"
#include "clif/python/stltypes.h"
#include "clif/python/slots.h"

namespace third__party_nucleus_io_python_tfrecord__writer_clifwrap {

using namespace clif;

static const char* ThisModuleName = "third_party.nucleus.io.python.tfrecord_writer";

#define _0 py::postconv::PASS
#define _1 UnicodeFromBytes
#define _2 UnicodeFromBytes

namespace pyTFRecordWriter {

struct wrapper {
  PyObject_HEAD
  ::clif::Instance<::nucleus::TFRecordWriter> cpp;
  PyObject* instance_dict = nullptr;
  PyObject* weakrefs = nullptr;
};

static ::nucleus::TFRecordWriter* ThisPtr(PyObject*);

// @classmethod from_file(filename:str, compression_type:str) -> TFRecordWriter
static PyObject* wrapNew_as_from_file(PyObject* cls, PyObject* args, PyObject* kw) {
  PyObject* a[2];
  const char* names[] = {
      "filename",
      "compression_type",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO:from_file", const_cast<char**>(names), &a[0], &a[1])) return nullptr;
  ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>> arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("from_file", names[0], "::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>", a[0]);
  ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>> arg2;
  if (!Clif_PyObjAs(a[1], &arg2)) return ArgError("from_file", names[1], "::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>", a[1]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::std::unique_ptr<::nucleus::TFRecordWriter> ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::TFRecordWriter::New(std::move(arg1), std::move(arg2));
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

// write(record:str) -> bool
static PyObject* wrapWriteRecord_as_write(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "record",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:write", const_cast<char**>(names), &a[0])) return nullptr;
  ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>> arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("write", names[0], "::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>", a[0]);
  // Call actual C++ method.
  ::nucleus::TFRecordWriter* c = ThisPtr(self);
  if (!c) return nullptr;
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  bool ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = c->WriteRecord(std::move(arg1));
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

// flush() -> bool
static PyObject* wrapFlush_as_flush(PyObject* self) {
  // Call actual C++ method.
  ::nucleus::TFRecordWriter* c = ThisPtr(self);
  if (!c) return nullptr;
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  bool ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = c->Flush();
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
  return Clif_PyObjFrom(std::move(ret0), {});
}

// close() -> bool
static PyObject* wrapClose_as_close(PyObject* self) {
  // Call actual C++ method.
  ::nucleus::TFRecordWriter* c = ThisPtr(self);
  if (!c) return nullptr;
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  bool ret0;
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
  return Clif_PyObjFrom(std::move(ret0), {});
}

static PyMethodDef MethodsStaticAlloc[] = {
  {"from_file", (PyCFunction)wrapNew_as_from_file, METH_VARARGS | METH_KEYWORDS | METH_CLASS, "from_file(filename:str, compression_type:str) -> TFRecordWriter\n  Calls C++ function\n  ::std::unique_ptr<::nucleus::TFRecordWriter> ::nucleus::TFRecordWriter::New(::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>)"},
  {"write", (PyCFunction)wrapWriteRecord_as_write, METH_VARARGS | METH_KEYWORDS, "write(record:str) -> bool\n  Calls C++ function\n  bool ::nucleus::TFRecordWriter::WriteRecord(::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>)"},
  {"flush", (PyCFunction)wrapFlush_as_flush, METH_NOARGS, "flush() -> bool\n  Calls C++ function\n  bool ::nucleus::TFRecordWriter::Flush()"},
  {"close", (PyCFunction)wrapClose_as_close, METH_NOARGS, "close() -> bool\n  Calls C++ function\n  bool ::nucleus::TFRecordWriter::Close()"},
  {"__reduce_ex__", (PyCFunction)::clif::ReduceExImpl, METH_VARARGS | METH_KEYWORDS, "Helper for pickle."},
  {}
};

// TFRecordWriter __new__
static PyObject* _new(PyTypeObject* type, Py_ssize_t nitems);

// TFRecordWriter __del__
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
      "TFRecordWriter");
  Py_INCREF(heap_type->ht_qualname);
  heap_type->ht_name = heap_type->ht_qualname;
  PyTypeObject *ty = &heap_type->ht_type;
  ty->tp_as_number = &heap_type->as_number;
  ty->tp_as_sequence = &heap_type->as_sequence;
  ty->tp_as_mapping = &heap_type->as_mapping;
#if PY_VERSION_HEX >= 0x03050000
  ty->tp_as_async = &heap_type->as_async;
#endif
  ty->tp_name = "third_party.nucleus.io.python.tfrecord_writer.TFRecordWriter";
  ty->tp_basicsize = sizeof(wrapper);
  ty->tp_dealloc = _dtor;
  ty->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
  ty->tp_doc = "CLIF wrapper for ::nucleus::TFRecordWriter";
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

static ::nucleus::TFRecordWriter* ThisPtr(PyObject* py) {
  if (Py_TYPE(py) == wrapper_Type) {
    return ::clif::python::Get(reinterpret_cast<wrapper*>(py)->cpp);
  }
  PyObject* base = PyObject_CallMethod(py, "as_nucleus_TFRecordWriter", nullptr);
  if (base == nullptr) {
    PyErr_Clear();
  } else {
    if (PyCapsule_CheckExact(base)) {
      void* p = PyCapsule_GetPointer(base, "::nucleus::TFRecordWriter");
      if (!PyErr_Occurred()) {
        ::nucleus::TFRecordWriter* c = static_cast<::nucleus::TFRecordWriter*>(p);
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
    PyErr_Format(PyExc_ValueError, "can't convert %s %s to ::nucleus::TFRecordWriter*", ClassName(py), ClassType(py));
  } else {
    PyErr_Format(PyExc_TypeError, "expecting %s instance, got %s %s", wrapper_Type->tp_name, ClassName(py), ClassType(py));
  }
  return nullptr;
}

}  // namespace pyTFRecordWriter

// Initialize module

bool Ready() {
  pyTFRecordWriter::wrapper_Type =
  pyTFRecordWriter::_build_heap_type();
  if (PyType_Ready(pyTFRecordWriter::wrapper_Type) < 0) return false;
  PyObject *modname = PyUnicode_FromString(ThisModuleName);
  if (modname == nullptr) return false;
  PyObject_SetAttrString((PyObject *) pyTFRecordWriter::wrapper_Type, "__module__", modname);
  Py_INCREF(pyTFRecordWriter::wrapper_Type);  // For PyModule_AddObject to steal.
  return true;
}

static struct PyModuleDef Module = {
  PyModuleDef_HEAD_INIT,
  ThisModuleName,
  "CLIF-generated module for third_party/nucleus/io/python/tfrecord_writer.clif", // module doc
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
  if (PyModule_AddObject(module, "TFRecordWriter", reinterpret_cast<PyObject*>(pyTFRecordWriter::wrapper_Type)) < 0) goto err;
  return module;
err:
  Py_DECREF(module);
  return nullptr;
}

}  // namespace third__party_nucleus_io_python_tfrecord__writer_clifwrap

namespace nucleus {
using namespace ::clif;
using ::clif::Clif_PyObjAs;
using ::clif::Clif_PyObjFrom;

// TFRecordWriter to/from ::nucleus::TFRecordWriter conversion

bool Clif_PyObjAs(PyObject* py, ::nucleus::TFRecordWriter** c) {
  CHECK(c != nullptr);
  if (Py_None == py) {
    *c = nullptr;
    return true;
  }
  ::nucleus::TFRecordWriter* cpp = third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = cpp;
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::shared_ptr<::nucleus::TFRecordWriter>* c) {
  CHECK(c != nullptr);
  ::nucleus::TFRecordWriter* cpp = third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = ::clif::MakeStdShared(reinterpret_cast<third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::wrapper*>(py)->cpp, cpp);
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::unique_ptr<::nucleus::TFRecordWriter>* c) {
  CHECK(c != nullptr);
  ::nucleus::TFRecordWriter* cpp = third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::ThisPtr(py);
  if (cpp == nullptr) return false;
  if (!reinterpret_cast<third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::wrapper*>(py)->cpp.Detach()) {
    PyErr_SetString(PyExc_ValueError, "Cannot convert TFRecordWriter instance to std::unique_ptr.");
    return false;
  }
  c->reset(cpp);
  return true;
}

PyObject* Clif_PyObjFrom(::nucleus::TFRecordWriter* c, py::PostConv unused) {
  CHECK(third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::wrapper_Type != nullptr) <<
    "---> Function Clif_PyObjFrom(::nucleus::TFRecordWriter) called before " <<
    third__party_nucleus_io_python_tfrecord__writer_clifwrap::ThisModuleName  <<
    " was imported from Python.";
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::wrapper_Type, NULL, NULL);
  reinterpret_cast<third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::wrapper*>(py)->cpp = ::clif::Instance<::nucleus::TFRecordWriter>(c, ::clif::UnOwnedResource());
  return py;
}

PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::TFRecordWriter> c, py::PostConv unused) {
  CHECK(third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::wrapper_Type != nullptr) <<
    "---> Function Clif_PyObjFrom(::nucleus::TFRecordWriter) called before " <<
    third__party_nucleus_io_python_tfrecord__writer_clifwrap::ThisModuleName  <<
    " was imported from Python.";
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::wrapper_Type, NULL, NULL);
  reinterpret_cast<third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::wrapper*>(py)->cpp = ::clif::Instance<::nucleus::TFRecordWriter>(c);
  return py;
}

PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::TFRecordWriter> c, py::PostConv unused) {
  CHECK(third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::wrapper_Type != nullptr) <<
    "---> Function Clif_PyObjFrom(::nucleus::TFRecordWriter) called before " <<
    third__party_nucleus_io_python_tfrecord__writer_clifwrap::ThisModuleName  <<
    " was imported from Python.";
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::wrapper_Type, NULL, NULL);
  reinterpret_cast<third__party_nucleus_io_python_tfrecord__writer_clifwrap::pyTFRecordWriter::wrapper*>(py)->cpp = ::clif::Instance<::nucleus::TFRecordWriter>(std::move(c));
  return py;
}

}  // namespace nucleus
