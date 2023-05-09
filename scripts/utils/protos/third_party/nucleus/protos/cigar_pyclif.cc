//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/protos/cigar.proto

#include "clif/python/runtime.h"
#include "clif/python/types.h"
#include "third_party/nucleus/protos/cigar_pyclif.h"

namespace nucleus { namespace genomics { namespace v1 {
using namespace ::clif;
using ::clif::Clif_PyObjAs;
using ::clif::Clif_PyObjFrom;

// CigarUnit to/from ::nucleus::genomics::v1::CigarUnit conversion

bool Clif_PyObjAs(PyObject* py, ::nucleus::genomics::v1::CigarUnit* c) {
  CHECK(c != nullptr);
  PyObject* type = ImportFQName("third_party.nucleus.protos.cigar_pb2.CigarUnit");
  if (!::clif::proto::TypeCheck(py, type, "", "CigarUnit") ) {
    return ::clif::proto::InGeneratedPool(py, c);
  }
  if (const proto2::Message* cpb = ::clif::proto::GetCProto(py)) {
    c->CopyFrom(*cpb);
    return true;
  }
  PyObject* ser = ::clif::proto::Serialize(py);
  if (ser == nullptr) return false;
  bool ok = c->ParsePartialFromArray(PyBytes_AS_STRING(ser), static_cast<int>(PyBytes_GET_SIZE(ser)));
  Py_DECREF(ser);
  if (!ok) PyErr_SetString(PyExc_ValueError, "Serialized bytes can't be parsed into C++ proto");
  return ok;
}

bool Clif_PyObjAs(PyObject* py, std::unique_ptr<::nucleus::genomics::v1::CigarUnit>* c) {
  CHECK(c != nullptr);
  if (!*c) c->reset(new ::nucleus::genomics::v1::CigarUnit);
  return Clif_PyObjAs(py, c->get());
}

PyObject* Clif_PyObjFrom(const ::nucleus::genomics::v1::CigarUnit& c, py::PostConv) {
  PyObject* type = ImportFQName("third_party.nucleus.protos.cigar_pb2.CigarUnit");
  return ::clif::proto::PyProtoFrom(&c, type, "");
}

PyObject* Clif_PyObjFrom(std::unique_ptr<const ::nucleus::genomics::v1::CigarUnit> c, py::PostConv) {
  if (!c) Py_RETURN_NONE;
  PyObject* type = ImportFQName("third_party.nucleus.protos.cigar_pb2.CigarUnit");
  return ::clif::proto::PyProtoFrom(c.get(), type, "");
}

PyObject* Clif_PyObjFrom(std::shared_ptr<const ::nucleus::genomics::v1::CigarUnit> c, py::PostConv) {
  if (!c) Py_RETURN_NONE;
  PyObject* type = ImportFQName("third_party.nucleus.protos.cigar_pb2.CigarUnit");
  return ::clif::proto::PyProtoFrom(c.get(), type, "");
}

// CigarUnit.Operation to/from enum ::nucleus::genomics::v1::CigarUnit::Operation conversion
bool Clif_PyObjAs(PyObject* py, ::nucleus::genomics::v1::CigarUnit::Operation* c) {
  CHECK(c != nullptr);
  int v;
  if (!Clif_PyObjAs(py, &v)) return false;
  *c = static_cast<::nucleus::genomics::v1::CigarUnit::Operation>(v);
  return true;
}
PyObject* Clif_PyObjFrom(::nucleus::genomics::v1::CigarUnit::Operation c, py::PostConv pc) {
  return Clif_PyObjFrom(static_cast<int>(c), pc);
}

} } }  // namespace nucleus::genomics::v1
