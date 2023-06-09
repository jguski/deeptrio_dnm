//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/vcf_writer.clif

#include <memory>
#include "absl/types/optional.h"
#include "third_party/nucleus/io/vcf_writer.h"
#include "clif/python/postconv.h"

namespace nucleus {
using namespace ::clif;

// CLIF use `::nucleus::VcfWriter` as VcfWriter
bool Clif_PyObjAs(PyObject* input, ::nucleus::VcfWriter** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::VcfWriter>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::VcfWriter>* output);
PyObject* Clif_PyObjFrom(::nucleus::VcfWriter*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::VcfWriter>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::VcfWriter>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::VcfWriter>::value>::type Clif_PyObjFrom(const ::nucleus::VcfWriter*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::VcfWriter>::value>::type Clif_PyObjFrom(const ::nucleus::VcfWriter&, py::PostConv) = delete;

}  // namespace nucleus

// CLIF init_module if (PyObject* m = PyImport_ImportModule("third_party.nucleus.io.python.vcf_writer")) Py_DECREF(m);
// CLIF init_module else goto err;
