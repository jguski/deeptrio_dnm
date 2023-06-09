//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/util/python/math.clif

#include <Python.h>
#include "absl/memory/memory.h"
#include "absl/types/optional.h"
#include "clif/python/types.h"
#include "math.h"
#include "clif/python/stltypes.h"
#include "clif/python/slots.h"

namespace third__party_nucleus_util_python_math_clifwrap {

using namespace clif;

static const char* ThisModuleName = "third_party.nucleus.util.python.math";

#define _0 py::postconv::PASS
#define _1 UnicodeFromBytes
#define _2 UnicodeFromBytes

// log10_ptrue_to_phred(log10_ptrue:float, value_if_not_finite:float) -> float
static PyObject* wrapLog10PTrueToPhred_as_log10_ptrue_to_phred(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[2];
  const char* names[] = {
      "log10_ptrue",
      "value_if_not_finite",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "OO:log10_ptrue_to_phred", const_cast<char**>(names), &a[0], &a[1])) return nullptr;
  double arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("log10_ptrue_to_phred", names[0], "double", a[0]);
  double arg2;
  if (!Clif_PyObjAs(a[1], &arg2)) return ArgError("log10_ptrue_to_phred", names[1], "double", a[1]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  double ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::Log10PTrueToPhred(std::move(arg1), std::move(arg2));
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

// phred_to_perror(phred:int) -> float
static PyObject* wrapPhredToPError_as_phred_to_perror(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "phred",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:phred_to_perror", const_cast<char**>(names), &a[0])) return nullptr;
  int arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("phred_to_perror", names[0], "int", a[0]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  double ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::PhredToPError(std::move(arg1));
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

// phred_to_log10_perror(phred:int) -> float
static PyObject* wrapPhredToLog10PError_as_phred_to_log10_perror(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "phred",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:phred_to_log10_perror", const_cast<char**>(names), &a[0])) return nullptr;
  int arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("phred_to_log10_perror", names[0], "int", a[0]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  double ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::PhredToLog10PError(std::move(arg1));
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

// perror_to_log10_perror(perror:float) -> float
static PyObject* wrapPErrorToLog10PError_as_perror_to_log10_perror(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "perror",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:perror_to_log10_perror", const_cast<char**>(names), &a[0])) return nullptr;
  double arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("perror_to_log10_perror", names[0], "double", a[0]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  double ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::PErrorToLog10PError(std::move(arg1));
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

// perror_to_phred(perror:float) -> float
static PyObject* wrapPErrorToPhred_as_perror_to_phred(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "perror",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:perror_to_phred", const_cast<char**>(names), &a[0])) return nullptr;
  double arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("perror_to_phred", names[0], "double", a[0]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  double ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::PErrorToPhred(std::move(arg1));
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

// log10_perror_to_phred(log10_perror:float) -> float
static PyObject* wrapLog10PErrorToPhred_as_log10_perror_to_phred(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "log10_perror",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:log10_perror_to_phred", const_cast<char**>(names), &a[0])) return nullptr;
  double arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("log10_perror_to_phred", names[0], "double", a[0]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  double ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::Log10PErrorToPhred(std::move(arg1));
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

// perror_to_rounded_phred(perror:float) -> int
static PyObject* wrapPErrorToRoundedPhred_as_perror_to_rounded_phred(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "perror",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:perror_to_rounded_phred", const_cast<char**>(names), &a[0])) return nullptr;
  double arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("perror_to_rounded_phred", names[0], "double", a[0]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  int ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::PErrorToRoundedPhred(std::move(arg1));
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

// log10_perror_to_rounded_phred(log10_perror:float) -> int
static PyObject* wrapLog10PErrorToRoundedPhred_as_log10_perror_to_rounded_phred(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "log10_perror",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:log10_perror_to_rounded_phred", const_cast<char**>(names), &a[0])) return nullptr;
  double arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("log10_perror_to_rounded_phred", names[0], "double", a[0]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  int ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::Log10PErrorToRoundedPhred(std::move(arg1));
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

// log10_perror_to_perror(log10_perror:float) -> float
static PyObject* wrapLog10ToReal_as_log10_perror_to_perror(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "log10_perror",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:log10_perror_to_perror", const_cast<char**>(names), &a[0])) return nullptr;
  double arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("log10_perror_to_perror", names[0], "double", a[0]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  double ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::Log10ToReal(std::move(arg1));
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

// zero_shift_log10_probs(log10_probs:list<float>) -> list<float>
static PyObject* wrapZeroShiftLikelihoods_as_zero_shift_log10_probs(PyObject* self, PyObject* args, PyObject* kw) {
  PyObject* a[1];
  const char* names[] = {
      "log10_probs",
      nullptr
  };
  if (!PyArg_ParseTupleAndKeywords(args, kw, "O:zero_shift_log10_probs", const_cast<char**>(names), &a[0])) return nullptr;
  ::std::vector<double> arg1;
  if (!Clif_PyObjAs(a[0], &arg1)) return ArgError("zero_shift_log10_probs", names[0], "::std::vector<double>", a[0]);
  // Call actual C++ method.
  Py_INCREF(args);
  Py_XINCREF(kw);
  PyThreadState* _save;
  Py_UNBLOCK_THREADS
  ::std::vector<double> ret0;
  PyObject* err_type = nullptr;
  std::string err_msg{"C++ exception"};
  try {
    ret0 = ::nucleus::ZeroShiftLikelihoods(std::move(arg1));
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
  {"log10_ptrue_to_phred", (PyCFunction)wrapLog10PTrueToPhred_as_log10_ptrue_to_phred, METH_VARARGS | METH_KEYWORDS, "log10_ptrue_to_phred(log10_ptrue:float, value_if_not_finite:float) -> float\n  Calls C++ function\n  double ::nucleus::Log10PTrueToPhred(double, double)"},
  {"phred_to_perror", (PyCFunction)wrapPhredToPError_as_phred_to_perror, METH_VARARGS | METH_KEYWORDS, "phred_to_perror(phred:int) -> float\n  Calls C++ function\n  double ::nucleus::PhredToPError(int)"},
  {"phred_to_log10_perror", (PyCFunction)wrapPhredToLog10PError_as_phred_to_log10_perror, METH_VARARGS | METH_KEYWORDS, "phred_to_log10_perror(phred:int) -> float\n  Calls C++ function\n  double ::nucleus::PhredToLog10PError(int)"},
  {"perror_to_log10_perror", (PyCFunction)wrapPErrorToLog10PError_as_perror_to_log10_perror, METH_VARARGS | METH_KEYWORDS, "perror_to_log10_perror(perror:float) -> float\n  Calls C++ function\n  double ::nucleus::PErrorToLog10PError(double)"},
  {"perror_to_phred", (PyCFunction)wrapPErrorToPhred_as_perror_to_phred, METH_VARARGS | METH_KEYWORDS, "perror_to_phred(perror:float) -> float\n  Calls C++ function\n  double ::nucleus::PErrorToPhred(double)"},
  {"log10_perror_to_phred", (PyCFunction)wrapLog10PErrorToPhred_as_log10_perror_to_phred, METH_VARARGS | METH_KEYWORDS, "log10_perror_to_phred(log10_perror:float) -> float\n  Calls C++ function\n  double ::nucleus::Log10PErrorToPhred(double)"},
  {"perror_to_rounded_phred", (PyCFunction)wrapPErrorToRoundedPhred_as_perror_to_rounded_phred, METH_VARARGS | METH_KEYWORDS, "perror_to_rounded_phred(perror:float) -> int\n  Calls C++ function\n  int ::nucleus::PErrorToRoundedPhred(double)"},
  {"log10_perror_to_rounded_phred", (PyCFunction)wrapLog10PErrorToRoundedPhred_as_log10_perror_to_rounded_phred, METH_VARARGS | METH_KEYWORDS, "log10_perror_to_rounded_phred(log10_perror:float) -> int\n  Calls C++ function\n  int ::nucleus::Log10PErrorToRoundedPhred(double)"},
  {"log10_perror_to_perror", (PyCFunction)wrapLog10ToReal_as_log10_perror_to_perror, METH_VARARGS | METH_KEYWORDS, "log10_perror_to_perror(log10_perror:float) -> float\n  Calls C++ function\n  double ::nucleus::Log10ToReal(double)"},
  {"zero_shift_log10_probs", (PyCFunction)wrapZeroShiftLikelihoods_as_zero_shift_log10_probs, METH_VARARGS | METH_KEYWORDS, "zero_shift_log10_probs(log10_probs:list<float>) -> list<float>\n  Calls C++ function\n  ::std::vector<double> ::nucleus::ZeroShiftLikelihoods(::std::vector<double>)"},
  {}
};

bool Ready() {
  return true;
}

static struct PyModuleDef Module = {
  PyModuleDef_HEAD_INIT,
  ThisModuleName,
  "CLIF-generated module for third_party/nucleus/util/python/math.clif", // module doc
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

}  // namespace third__party_nucleus_util_python_math_clifwrap
