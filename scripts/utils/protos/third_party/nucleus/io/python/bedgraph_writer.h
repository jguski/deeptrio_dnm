//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/bedgraph_writer.clif

#include <memory>
#include "absl/types/optional.h"
#include "third_party/nucleus/io/bedgraph_writer.h"
#include "clif/python/postconv.h"

namespace nucleus {
using namespace ::clif;

// CLIF use `::nucleus::BedGraphWriter` as BedGraphWriter
bool Clif_PyObjAs(PyObject* input, ::nucleus::BedGraphWriter** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::BedGraphWriter>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::BedGraphWriter>* output);
PyObject* Clif_PyObjFrom(::nucleus::BedGraphWriter*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::BedGraphWriter>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::BedGraphWriter>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::BedGraphWriter>::value>::type Clif_PyObjFrom(const ::nucleus::BedGraphWriter*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::BedGraphWriter>::value>::type Clif_PyObjFrom(const ::nucleus::BedGraphWriter&, py::PostConv) = delete;

}  // namespace nucleus

// CLIF init_module if (PyObject* m = PyImport_ImportModule("third_party.nucleus.io.python.bedgraph_writer")) Py_DECREF(m);
// CLIF init_module else goto err;