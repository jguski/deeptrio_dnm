//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/bed_writer.clif

#include <memory>
#include "absl/types/optional.h"
#include "third_party/nucleus/io/bed_writer.h"
#include "clif/python/postconv.h"

namespace nucleus {
using namespace ::clif;

// CLIF use `::nucleus::BedWriter` as BedWriter
bool Clif_PyObjAs(PyObject* input, ::nucleus::BedWriter** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::BedWriter>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::BedWriter>* output);
PyObject* Clif_PyObjFrom(::nucleus::BedWriter*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::BedWriter>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::BedWriter>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::BedWriter>::value>::type Clif_PyObjFrom(const ::nucleus::BedWriter*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::BedWriter>::value>::type Clif_PyObjFrom(const ::nucleus::BedWriter&, py::PostConv) = delete;

}  // namespace nucleus

// CLIF init_module if (PyObject* m = PyImport_ImportModule("third_party.nucleus.io.python.bed_writer")) Py_DECREF(m);
// CLIF init_module else goto err;