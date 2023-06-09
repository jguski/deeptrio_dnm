//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/sam_reader.clif

#include <Python.h>
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "clif/python/types.h"
#include "third_party/nucleus/protos/range_pyclif.h"
#include "third_party/nucleus/protos/reads_pyclif.h"
#include "third_party/nucleus/protos/reference_pyclif.h"
#include "third_party/nucleus/util/proto_clif_converter.h"
#include "third_party/nucleus/vendor/statusor_clif_converters.h"
#include "sam_reader.h"
#include "clif/python/stltypes.h"
#include "clif/python/slots.h"

namespace third__party_nucleus_io_python_sam__reader_clifwrap {

using namespace clif;

static const char* ThisModuleName = "third_party.nucleus.io.python.sam_reader";

#define _0 py::postconv::PASS
#define _1 UnicodeFromBytes
#define _2 UnicodeFromBytes

namespace pySamIterable {

struct wrapper {
  PyObject_HEAD
  ::clif::Instance<::nucleus::Iterable< ::nucleus::genomics::v1::Read>> cpp;
  PyObject* instance_dict = nullptr;
  PyObject* weakrefs = nullptr;
};

static ::nucleus::Iterable< ::nucleus::genomics::v1::Read>* ThisPtr(PyObject*);

// PythonNext(read:EmptyProtoPtr<Read>) -> StatusOr<bool>
static PyObject* wrapPythonNext(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "read",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:PythonNext", const_cast<char**>(names), &a[0])) return nullptr;
  ::nucleus::EmptyProtoPtr< ::nucleus::genomics::v1::Read> arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("PythonNext", names[0], "::nucleus::EmptyProtoPtr< ::nucleus::genomics::v1::Read>", a[0]);
  // Call actual C++ method.
  ::nucleus::Iterable< ::nucleus::genomics::v1::Read>* c = ThisPtr(self);
  if (!c) return nullptr;
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::nucleus::StatusOr<bool> ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = c->PythonNext(std::move(arg1));
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

// Release() -> Status
static PyObject* wrapRelease(PyObject* self) {
  // Call actual C++ method.
  ::nucleus::Iterable< ::nucleus::genomics::v1::Read>* c = ThisPtr(self);
  if (!c) return nullptr;
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::tensorflow::Status ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = c->Release();
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

// __enter__@() -> Status
static PyObject* wrapPythonEnter_as___enter__(PyObject* self) {
  // Call actual C++ method.
  ::nucleus::Iterable< ::nucleus::genomics::v1::Read>* c = ThisPtr(self);
  if (!c) return nullptr;
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::tensorflow::Status ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = c->PythonEnter();
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
  Py_INCREF(self);
  return self;
}

// __exit__@() -> Status
static PyObject* wrapPythonExit_as___exit__(PyObject* self) {
  // Call actual C++ method.
  ::nucleus::Iterable< ::nucleus::genomics::v1::Read>* c = ThisPtr(self);
  if (!c) return nullptr;
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::tensorflow::Status ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = c->PythonExit();
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

// Implicit cast this as ::nucleus::IterableBase*
static PyObject* as_nucleus_IterableBase(PyObject* self) {
  ::nucleus::IterableBase* p = ::clif::python::Get(reinterpret_cast<wrapper*>(self)->cpp);
  if (p == nullptr) return nullptr;
  return PyCapsule_New(p, "::nucleus::IterableBase", nullptr);
}

static PyMethodDef MethodsStaticAlloc[] = {
  {"PythonNext", (PyCFunction)wrapPythonNext, METH_VARARGS | METH_KEYWORDS, "PythonNext(read:EmptyProtoPtr<Read>) -> StatusOr<bool>\n  Calls C++ function\n  ::nucleus::StatusOr<bool> ::nucleus::Iterable<nucleus::genomics::v1::Read>::PythonNext(::nucleus::EmptyProtoPtr< ::nucleus::genomics::v1::Read>)"},
  {"Release", (PyCFunction)wrapRelease, METH_NOARGS, "Release() -> Status\n  Calls C++ function\n  ::tensorflow::Status ::nucleus::IterableBase::Release()"},
  {"__enter__", (PyCFunction)wrapPythonEnter_as___enter__, METH_NOARGS, "__enter__@() -> Status\n  Calls C++ function\n  ::tensorflow::Status ::nucleus::IterableBase::PythonEnter()"},
  {"__exit__", (PyCFunction)wrapPythonExit_as___exit__, METH_VARARGS | METH_KEYWORDS, "__exit__@() -> Status\n  Calls C++ function\n  ::tensorflow::Status ::nucleus::IterableBase::PythonExit()"},
  {"as_nucleus_IterableBase", (PyCFunction)as_nucleus_IterableBase, METH_NOARGS, "Upcast to ::nucleus::IterableBase*"},
  {"__reduce_ex__", (PyCFunction)::clif::ReduceExImpl, METH_VARARGS | METH_KEYWORDS, "Helper for pickle."},
  {}
};

// SamIterable __new__
static PyObject* _new(PyTypeObject* type, Py_ssize_t nitems);

// SamIterable __del__
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
      "SamIterable");
  Py_INCREF(heap_type->ht_qualname);
  heap_type->ht_name = heap_type->ht_qualname;
  PyTypeObject *ty = &heap_type->ht_type;
  ty->tp_as_number = &heap_type->as_number;
  ty->tp_as_sequence = &heap_type->as_sequence;
  ty->tp_as_mapping = &heap_type->as_mapping;
#if PY_VERSION_HEX >= 0x03050000
  ty->tp_as_async = &heap_type->as_async;
#endif
  ty->tp_name = "third_party.nucleus.io.python.sam_reader.SamIterable";
  ty->tp_basicsize = sizeof(wrapper);
  ty->tp_dealloc = _dtor;
  ty->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_IS_ABSTRACT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
  ty->tp_doc = "CLIF wrapper for ::nucleus::Iterable< ::nucleus::genomics::v1::Read>";
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

static ::nucleus::Iterable< ::nucleus::genomics::v1::Read>* ThisPtr(PyObject* py) {
  if (Py_TYPE(py) == wrapper_Type) {
    return ::clif::python::Get(reinterpret_cast<wrapper*>(py)->cpp);
  }
  PyObject* base = PyObject_CallMethod(py, "as_nucleus_Iterable__nucleus_genomics_v1_Read", nullptr);
  if (base == nullptr) {
    PyErr_Clear();
  } else {
    if (PyCapsule_CheckExact(base)) {
      void* p = PyCapsule_GetPointer(base, "::nucleus::Iterable< ::nucleus::genomics::v1::Read>");
      if (!PyErr_Occurred()) {
        ::nucleus::Iterable< ::nucleus::genomics::v1::Read>* c = static_cast<::nucleus::Iterable< ::nucleus::genomics::v1::Read>*>(p);
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
    PyErr_Format(PyExc_ValueError, "can't convert %s %s to ::nucleus::Iterable< ::nucleus::genomics::v1::Read>*", ClassName(py), ClassType(py));
  } else {
    PyErr_Format(PyExc_TypeError, "expecting %s instance, got %s %s", wrapper_Type->tp_name, ClassName(py), ClassType(py));
  }
  return nullptr;
}

}  // namespace pySamIterable

namespace pySamReader {

struct wrapper {
  PyObject_HEAD
  ::clif::Instance<::nucleus::SamReader> cpp;
  PyObject* instance_dict = nullptr;
  PyObject* weakrefs = nullptr;
};

static ::nucleus::SamReader* ThisPtr(PyObject*);

// @classmethod from_file(reads_path:str, ref_path:str, options:SamReaderOptions) -> StatusOr<SamReader>
static PyObject* wrapFromFile_as_from_file(PyObject* cls, PyObject* args, PyObject* kw) {
  PyObject* a[3];
  const char* names[] = {
      "reads_path",
      "ref_path",
      "options",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO:from_file", const_cast<char**>(names), &a[0], &a[1], &a[2])) return nullptr;
  ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>> arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("from_file", names[0], "::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>", a[0]);
  ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>> arg2;
  if (!Clif_PyObjAs(a[1], &arg2)) return ArgError("from_file", names[1], "::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>", a[1]);
  ::nucleus::genomics::v1::SamReaderOptions arg3;
  if (!Clif_PyObjAs(a[2], &arg3)) return ArgError("from_file", names[2], "::nucleus::genomics::v1::SamReaderOptions", a[2]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::nucleus::StatusOr< ::std::unique_ptr< ::nucleus::SamReader>> ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::SamReader::FromFile(std::move(arg1), std::move(arg2), std::move(arg3));
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

// iterate() -> StatusOr<SamIterable>
static PyObject* wrapIterate_as_iterate(PyObject* self) {
  // Call actual C++ method.
  ::nucleus::SamReader* c = ThisPtr(self);
  if (!c) return nullptr;
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::nucleus::StatusOr< ::std::shared_ptr< ::nucleus::SamIterable>> ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = c->Iterate();
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
  PyObject* pyproc = ImportFQName("third_party.nucleus.io.clif_postproc.WrappedSamIterable");
  if (pyproc == nullptr) {
    Py_DECREF(result_tuple);
    return nullptr;
  }
  p = PyObject_CallObject(pyproc, result_tuple);
  Py_DECREF(pyproc);
  Py_CLEAR(result_tuple);
  result_tuple = p;
  return result_tuple;
}

// query(region:Range) -> StatusOr<SamIterable>
static PyObject* wrapQuery_as_query(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "region",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:query", const_cast<char**>(names), &a[0])) return nullptr;
  ::nucleus::genomics::v1::Range arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("query", names[0], "::nucleus::genomics::v1::Range", a[0]);
  // Call actual C++ method.
  ::nucleus::SamReader* c = ThisPtr(self);
  if (!c) return nullptr;
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::nucleus::StatusOr< ::std::shared_ptr< ::nucleus::SamIterable>> ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = c->Query(std::move(arg1));
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
  // Convert return values to Python.
  PyObject* p, * result_tuple = PyTuple_New(1);
  if (result_tuple == nullptr) return nullptr;
  if ((p=Clif_PyObjFrom(std::move(ret0), {})) == nullptr) {
    Py_DECREF(result_tuple);
    return nullptr;
  }
  PyTuple_SET_ITEM(result_tuple, 0, p);
  PyObject* pyproc = ImportFQName("third_party.nucleus.io.clif_postproc.WrappedSamIterable");
  if (pyproc == nullptr) {
    Py_DECREF(result_tuple);
    return nullptr;
  }
  p = PyObject_CallObject(pyproc, result_tuple);
  Py_DECREF(pyproc);
  Py_CLEAR(result_tuple);
  result_tuple = p;
  return result_tuple;
}

static PyObject* get_header(PyObject* self, void* xdata) {
  auto cpp = ThisPtr(self); if (!cpp) return nullptr;
  return Clif_PyObjFrom(cpp->Header(), {});
}

// __enter__@() -> Status
static PyObject* wrapPythonEnter_as___enter__(PyObject* self) {
  // Call actual C++ method.
  ::nucleus::SamReader* c = ThisPtr(self);
  if (!c) return nullptr;
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::tensorflow::Status ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = c->PythonEnter();
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
  Py_INCREF(self);
  return self;
}

// __exit__@() -> Status
static PyObject* wrapClose_as___exit__(PyObject* self) {
  // Call actual C++ method.
  ::nucleus::SamReader* c = ThisPtr(self);
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

static PyGetSetDef Properties[] = {
  {"header", get_header, nullptr, "C++ clif_type_19 SamReader.Header()"},
  {}
};

// Implicit cast this as ::nucleus::Reader*
static PyObject* as_nucleus_Reader(PyObject* self) {
  ::nucleus::Reader* p = ::clif::python::Get(reinterpret_cast<wrapper*>(self)->cpp);
  if (p == nullptr) return nullptr;
  return PyCapsule_New(p, "::nucleus::Reader", nullptr);
}

static PyMethodDef MethodsStaticAlloc[] = {
  {"from_file", (PyCFunction)wrapFromFile_as_from_file, METH_VARARGS | METH_KEYWORDS | METH_CLASS, "from_file(reads_path:str, ref_path:str, options:SamReaderOptions) -> StatusOr<SamReader>\n  Calls C++ function\n  ::nucleus::StatusOr< ::std::unique_ptr< ::nucleus::SamReader>> ::nucleus::SamReader::FromFile(::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::std::basic_string<char, ::std::char_traits<char>, ::std::allocator<char>>, ::nucleus::genomics::v1::SamReaderOptions)"},
  {"iterate", (PyCFunction)wrapIterate_as_iterate, METH_NOARGS, "iterate() -> StatusOr<SamIterable>\n  Calls C++ function\n  ::nucleus::StatusOr< ::std::shared_ptr< ::nucleus::SamIterable>> ::nucleus::SamReader::Iterate()"},
  {"query", (PyCFunction)wrapQuery_as_query, METH_VARARGS | METH_KEYWORDS, "query(region:Range) -> StatusOr<SamIterable>\n  Calls C++ function\n  ::nucleus::StatusOr< ::std::shared_ptr< ::nucleus::SamIterable>> ::nucleus::SamReader::Query(::nucleus::genomics::v1::Range)"},
  {"__enter__", (PyCFunction)wrapPythonEnter_as___enter__, METH_NOARGS, "__enter__@() -> Status\n  Calls C++ function\n  ::tensorflow::Status ::nucleus::SamReader::PythonEnter()"},
  {"__exit__", (PyCFunction)wrapClose_as___exit__, METH_VARARGS | METH_KEYWORDS, "__exit__@() -> Status\n  Calls C++ function\n  ::tensorflow::Status ::nucleus::SamReader::Close()"},
  {"as_nucleus_Reader", (PyCFunction)as_nucleus_Reader, METH_NOARGS, "Upcast to ::nucleus::Reader*"},
  {"__reduce_ex__", (PyCFunction)::clif::ReduceExImpl, METH_VARARGS | METH_KEYWORDS, "Helper for pickle."},
  {}
};

// SamReader __new__
static PyObject* _new(PyTypeObject* type, Py_ssize_t nitems);

// SamReader __del__
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
      "SamReader");
  Py_INCREF(heap_type->ht_qualname);
  heap_type->ht_name = heap_type->ht_qualname;
  PyTypeObject *ty = &heap_type->ht_type;
  ty->tp_as_number = &heap_type->as_number;
  ty->tp_as_sequence = &heap_type->as_sequence;
  ty->tp_as_mapping = &heap_type->as_mapping;
#if PY_VERSION_HEX >= 0x03050000
  ty->tp_as_async = &heap_type->as_async;
#endif
  ty->tp_name = "third_party.nucleus.io.python.sam_reader.SamReader";
  ty->tp_basicsize = sizeof(wrapper);
  ty->tp_dealloc = _dtor;
  ty->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HEAPTYPE;
  ty->tp_doc = "CLIF wrapper for ::nucleus::SamReader";
  ty->tp_methods = MethodsStaticAlloc;
  ty->tp_getset = Properties;
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

static ::nucleus::SamReader* ThisPtr(PyObject* py) {
  if (Py_TYPE(py) == wrapper_Type) {
    return ::clif::python::Get(reinterpret_cast<wrapper*>(py)->cpp);
  }
  PyObject* base = PyObject_CallMethod(py, "as_nucleus_SamReader", nullptr);
  if (base == nullptr) {
    PyErr_Clear();
  } else {
    if (PyCapsule_CheckExact(base)) {
      void* p = PyCapsule_GetPointer(base, "::nucleus::SamReader");
      if (!PyErr_Occurred()) {
        ::nucleus::SamReader* c = static_cast<::nucleus::SamReader*>(p);
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
    PyErr_Format(PyExc_ValueError, "can't convert %s %s to ::nucleus::SamReader*", ClassName(py), ClassType(py));
  } else {
    PyErr_Format(PyExc_TypeError, "expecting %s instance, got %s %s", wrapper_Type->tp_name, ClassName(py), ClassType(py));
  }
  return nullptr;
}

}  // namespace pySamReader

// Initialize module

bool Ready() {
  pySamIterable::wrapper_Type =
  pySamIterable::_build_heap_type();
  if (PyType_Ready(pySamIterable::wrapper_Type) < 0) return false;
  PyObject *modname = PyUnicode_FromString(ThisModuleName);
  if (modname == nullptr) return false;
  PyObject_SetAttrString((PyObject *) pySamIterable::wrapper_Type, "__module__", modname);
  Py_INCREF(pySamIterable::wrapper_Type);  // For PyModule_AddObject to steal.
  pySamReader::wrapper_Type =
  pySamReader::_build_heap_type();
  if (PyType_Ready(pySamReader::wrapper_Type) < 0) return false;
  PyObject_SetAttrString((PyObject *) pySamReader::wrapper_Type, "__module__", modname);
  Py_INCREF(pySamReader::wrapper_Type);  // For PyModule_AddObject to steal.
  return true;
}

static struct PyModuleDef Module = {
  PyModuleDef_HEAD_INIT,
  ThisModuleName,
  "CLIF-generated module for third_party/nucleus/io/python/sam_reader.clif", // module doc
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
  if (PyModule_AddObject(module, "SamIterable", reinterpret_cast<PyObject*>(pySamIterable::wrapper_Type)) < 0) goto err;
  if (PyModule_AddObject(module, "SamReader", reinterpret_cast<PyObject*>(pySamReader::wrapper_Type)) < 0) goto err;
  return module;
err:
  Py_DECREF(module);
  return nullptr;
}

}  // namespace third__party_nucleus_io_python_sam__reader_clifwrap

namespace nucleus {
using namespace ::clif;
using ::clif::Clif_PyObjAs;
using ::clif::Clif_PyObjFrom;

// SamIterable to/from ::nucleus::Iterable< ::nucleus::genomics::v1::Read> conversion

bool Clif_PyObjAs(PyObject* py, ::nucleus::Iterable< ::nucleus::genomics::v1::Read>** c) {
  CHECK(c != nullptr);
  if (Py_None == py) {
    *c = nullptr;
    return true;
  }
  ::nucleus::Iterable< ::nucleus::genomics::v1::Read>* cpp = third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = cpp;
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::shared_ptr<::nucleus::Iterable< ::nucleus::genomics::v1::Read>>* c) {
  CHECK(c != nullptr);
  ::nucleus::Iterable< ::nucleus::genomics::v1::Read>* cpp = third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = ::clif::MakeStdShared(reinterpret_cast<third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::wrapper*>(py)->cpp, cpp);
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::unique_ptr<::nucleus::Iterable< ::nucleus::genomics::v1::Read>>* c) {
  CHECK(c != nullptr);
  ::nucleus::Iterable< ::nucleus::genomics::v1::Read>* cpp = third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::ThisPtr(py);
  if (cpp == nullptr) return false;
  if (!reinterpret_cast<third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::wrapper*>(py)->cpp.Detach()) {
    PyErr_SetString(PyExc_ValueError, "Cannot convert SamIterable instance to std::unique_ptr.");
    return false;
  }
  c->reset(cpp);
  return true;
}

PyObject* Clif_PyObjFrom(::nucleus::Iterable< ::nucleus::genomics::v1::Read>* c, py::PostConv unused) {
  CHECK(third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::wrapper_Type != nullptr) <<
    "---> Function Clif_PyObjFrom(::nucleus::Iterable< ::nucleus::genomics::v1::Read>) called before " <<
    third__party_nucleus_io_python_sam__reader_clifwrap::ThisModuleName  <<
    " was imported from Python.";
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::wrapper_Type, NULL, NULL);
  reinterpret_cast<third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::wrapper*>(py)->cpp = ::clif::Instance<::nucleus::Iterable< ::nucleus::genomics::v1::Read>>(c, ::clif::UnOwnedResource());
  return py;
}

PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::Iterable< ::nucleus::genomics::v1::Read>> c, py::PostConv unused) {
  CHECK(third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::wrapper_Type != nullptr) <<
    "---> Function Clif_PyObjFrom(::nucleus::Iterable< ::nucleus::genomics::v1::Read>) called before " <<
    third__party_nucleus_io_python_sam__reader_clifwrap::ThisModuleName  <<
    " was imported from Python.";
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::wrapper_Type, NULL, NULL);
  reinterpret_cast<third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::wrapper*>(py)->cpp = ::clif::Instance<::nucleus::Iterable< ::nucleus::genomics::v1::Read>>(c);
  return py;
}

PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::Iterable< ::nucleus::genomics::v1::Read>> c, py::PostConv unused) {
  CHECK(third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::wrapper_Type != nullptr) <<
    "---> Function Clif_PyObjFrom(::nucleus::Iterable< ::nucleus::genomics::v1::Read>) called before " <<
    third__party_nucleus_io_python_sam__reader_clifwrap::ThisModuleName  <<
    " was imported from Python.";
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::wrapper_Type, NULL, NULL);
  reinterpret_cast<third__party_nucleus_io_python_sam__reader_clifwrap::pySamIterable::wrapper*>(py)->cpp = ::clif::Instance<::nucleus::Iterable< ::nucleus::genomics::v1::Read>>(std::move(c));
  return py;
}

// SamReader to/from ::nucleus::SamReader conversion

bool Clif_PyObjAs(PyObject* py, ::nucleus::SamReader** c) {
  CHECK(c != nullptr);
  if (Py_None == py) {
    *c = nullptr;
    return true;
  }
  ::nucleus::SamReader* cpp = third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = cpp;
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::shared_ptr<::nucleus::SamReader>* c) {
  CHECK(c != nullptr);
  ::nucleus::SamReader* cpp = third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::ThisPtr(py);
  if (cpp == nullptr) return false;
  *c = ::clif::MakeStdShared(reinterpret_cast<third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::wrapper*>(py)->cpp, cpp);
  return true;
}

bool Clif_PyObjAs(PyObject* py, std::unique_ptr<::nucleus::SamReader>* c) {
  CHECK(c != nullptr);
  ::nucleus::SamReader* cpp = third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::ThisPtr(py);
  if (cpp == nullptr) return false;
  if (!reinterpret_cast<third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::wrapper*>(py)->cpp.Detach()) {
    PyErr_SetString(PyExc_ValueError, "Cannot convert SamReader instance to std::unique_ptr.");
    return false;
  }
  c->reset(cpp);
  return true;
}

PyObject* Clif_PyObjFrom(::nucleus::SamReader* c, py::PostConv unused) {
  CHECK(third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::wrapper_Type != nullptr) <<
    "---> Function Clif_PyObjFrom(::nucleus::SamReader) called before " <<
    third__party_nucleus_io_python_sam__reader_clifwrap::ThisModuleName  <<
    " was imported from Python.";
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::wrapper_Type, NULL, NULL);
  reinterpret_cast<third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::wrapper*>(py)->cpp = ::clif::Instance<::nucleus::SamReader>(c, ::clif::UnOwnedResource());
  return py;
}

PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::SamReader> c, py::PostConv unused) {
  CHECK(third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::wrapper_Type != nullptr) <<
    "---> Function Clif_PyObjFrom(::nucleus::SamReader) called before " <<
    third__party_nucleus_io_python_sam__reader_clifwrap::ThisModuleName  <<
    " was imported from Python.";
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::wrapper_Type, NULL, NULL);
  reinterpret_cast<third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::wrapper*>(py)->cpp = ::clif::Instance<::nucleus::SamReader>(c);
  return py;
}

PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::SamReader> c, py::PostConv unused) {
  CHECK(third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::wrapper_Type != nullptr) <<
    "---> Function Clif_PyObjFrom(::nucleus::SamReader) called before " <<
    third__party_nucleus_io_python_sam__reader_clifwrap::ThisModuleName  <<
    " was imported from Python.";
  if (c == nullptr) Py_RETURN_NONE;
  PyObject* py = PyType_GenericNew(third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::wrapper_Type, NULL, NULL);
  reinterpret_cast<third__party_nucleus_io_python_sam__reader_clifwrap::pySamReader::wrapper*>(py)->cpp = ::clif::Instance<::nucleus::SamReader>(std::move(c));
  return py;
}

}  // namespace nucleus
