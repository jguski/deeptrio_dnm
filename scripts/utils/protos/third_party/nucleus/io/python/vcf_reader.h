//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/vcf_reader.clif

#include <memory>
#include "absl/types/optional.h"
#include "third_party/nucleus/io/vcf_reader.h"
#include "clif/python/postconv.h"

namespace nucleus {
using namespace ::clif;

// CLIF use `::nucleus::Iterable< ::nucleus::genomics::v1::Variant>` as VariantIterable
bool Clif_PyObjAs(PyObject* input, ::nucleus::Iterable< ::nucleus::genomics::v1::Variant>** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::Iterable< ::nucleus::genomics::v1::Variant>>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::Iterable< ::nucleus::genomics::v1::Variant>>* output);
PyObject* Clif_PyObjFrom(::nucleus::Iterable< ::nucleus::genomics::v1::Variant>*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::Iterable< ::nucleus::genomics::v1::Variant>>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::Iterable< ::nucleus::genomics::v1::Variant>>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::Iterable< ::nucleus::genomics::v1::Variant>>::value>::type Clif_PyObjFrom(const ::nucleus::Iterable< ::nucleus::genomics::v1::Variant>*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::Iterable< ::nucleus::genomics::v1::Variant>>::value>::type Clif_PyObjFrom(const ::nucleus::Iterable< ::nucleus::genomics::v1::Variant>&, py::PostConv) = delete;
// CLIF use `::nucleus::VcfReader` as VcfReader
bool Clif_PyObjAs(PyObject* input, ::nucleus::VcfReader** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::VcfReader>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::VcfReader>* output);
PyObject* Clif_PyObjFrom(::nucleus::VcfReader*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::VcfReader>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::VcfReader>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::VcfReader>::value>::type Clif_PyObjFrom(const ::nucleus::VcfReader*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::VcfReader>::value>::type Clif_PyObjFrom(const ::nucleus::VcfReader&, py::PostConv) = delete;

}  // namespace nucleus

// CLIF init_module if (PyObject* m = PyImport_ImportModule("third_party.nucleus.io.python.vcf_reader")) Py_DECREF(m);
// CLIF init_module else goto err;
