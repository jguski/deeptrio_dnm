//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/io/python/tfrecord_reader.clif

#include <memory>
#include "absl/types/optional.h"
#include "third_party/nucleus/io/tfrecord_reader.h"
#include "clif/python/postconv.h"

namespace nucleus {
using namespace ::clif;

// CLIF use `::nucleus::TFRecordReader` as TFRecordReader
bool Clif_PyObjAs(PyObject* input, ::nucleus::TFRecordReader** output);
bool Clif_PyObjAs(PyObject* input, std::shared_ptr<::nucleus::TFRecordReader>* output);
bool Clif_PyObjAs(PyObject* input, std::unique_ptr<::nucleus::TFRecordReader>* output);
PyObject* Clif_PyObjFrom(::nucleus::TFRecordReader*, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<::nucleus::TFRecordReader>, py::PostConv);
PyObject* Clif_PyObjFrom(std::unique_ptr<::nucleus::TFRecordReader>, py::PostConv);
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::TFRecordReader>::value>::type Clif_PyObjFrom(const ::nucleus::TFRecordReader*, py::PostConv) = delete;
template<typename T>
typename std::enable_if<std::is_same<T, ::nucleus::TFRecordReader>::value>::type Clif_PyObjFrom(const ::nucleus::TFRecordReader&, py::PostConv) = delete;

}  // namespace nucleus

// CLIF init_module if (PyObject* m = PyImport_ImportModule("third_party.nucleus.io.python.tfrecord_reader")) Py_DECREF(m);
// CLIF init_module else goto err;