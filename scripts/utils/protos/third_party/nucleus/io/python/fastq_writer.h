//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/fastq_writer.clif

#include <memory>
#include "absl/types/optional.h"
#include "third_party/nucleus/io/fastq_writer.h"
#include "clif/python/postconv.h"

namespace nucleus {
using namespace ::clif;

// CLIF use `::nucleus::FastqWriter` as FastqWriter
bool Clif_PyObjAs(PyObject* input, ::nucleus::FastqWriter** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::FastqWriter>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::FastqWriter>* output);
PyObject* Clif_PyObjFrom(::nucleus::FastqWriter*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::FastqWriter>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::FastqWriter>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::FastqWriter>::value>::type Clif_PyObjFrom(const ::nucleus::FastqWriter*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::FastqWriter>::value>::type Clif_PyObjFrom(const ::nucleus::FastqWriter&, py::PostConv) = delete;

}  // namespace nucleus

// CLIF init_module if (PyObject* m = PyImport_ImportModule("third_party.nucleus.io.python.fastq_writer")) Py_DECREF(m);
// CLIF init_module else goto err;
