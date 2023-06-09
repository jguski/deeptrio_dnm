//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/gff_reader.clif

#include <memory>
#include "absl/types/optional.h"
#include "third_party/nucleus/io/gff_reader.h"
#include "clif/python/postconv.h"

namespace nucleus {
using namespace ::clif;

// CLIF use `::nucleus::Iterable< ::nucleus::genomics::v1::GffRecord>` as GffIterable
bool Clif_PyObjAs(PyObject* input, ::nucleus::Iterable< ::nucleus::genomics::v1::GffRecord>** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::Iterable< ::nucleus::genomics::v1::GffRecord>>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::Iterable< ::nucleus::genomics::v1::GffRecord>>* output);
PyObject* Clif_PyObjFrom(::nucleus::Iterable< ::nucleus::genomics::v1::GffRecord>*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::Iterable< ::nucleus::genomics::v1::GffRecord>>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::Iterable< ::nucleus::genomics::v1::GffRecord>>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::Iterable< ::nucleus::genomics::v1::GffRecord>>::value>::type Clif_PyObjFrom(const ::nucleus::Iterable< ::nucleus::genomics::v1::GffRecord>*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::Iterable< ::nucleus::genomics::v1::GffRecord>>::value>::type Clif_PyObjFrom(const ::nucleus::Iterable< ::nucleus::genomics::v1::GffRecord>&, py::PostConv) = delete;
// CLIF use `::nucleus::GffReader` as GffReader
bool Clif_PyObjAs(PyObject* input, ::nucleus::GffReader** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::GffReader>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::GffReader>* output);
PyObject* Clif_PyObjFrom(::nucleus::GffReader*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::GffReader>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::GffReader>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::GffReader>::value>::type Clif_PyObjFrom(const ::nucleus::GffReader*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::GffReader>::value>::type Clif_PyObjFrom(const ::nucleus::GffReader&, py::PostConv) = delete;

}  // namespace nucleus

// CLIF init_module if (PyObject* m = PyImport_ImportModule("third_party.nucleus.io.python.gff_reader")) Py_DECREF(m);
// CLIF init_module else goto err;
