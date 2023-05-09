//////////////////////////////////////////////////////////////////////
// This file was automatically generated by PyCLIF.
// Version 0.3
//////////////////////////////////////////////////////////////////////
// source: third_party/nucleus/protos/gff.proto

#include "third_party/nucleus/protos/gff.pb.h"
#include "clif/python/postconv.h"

namespace nucleus { namespace genomics { namespace v1 {
using namespace ::clif;

// CLIF use `::nucleus::genomics::v1::GffRecord` as GffRecord
bool Clif_PyObjAs(PyObject* input, ::nucleus::genomics::v1::GffRecord* output);
PyObject* Clif_PyObjFrom(const ::nucleus::genomics::v1::GffRecord&, py::PostConv);
bool Clif_PyObjAs(PyObject*, std::unique_ptr<::nucleus::genomics::v1::GffRecord>*);
PyObject* Clif_PyObjFrom(std::unique_ptr<const ::nucleus::genomics::v1::GffRecord>, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<const ::nucleus::genomics::v1::GffRecord>, py::PostConv);
// CLIF use `::nucleus::genomics::v1::GffHeader` as GffHeader
bool Clif_PyObjAs(PyObject* input, ::nucleus::genomics::v1::GffHeader* output);
PyObject* Clif_PyObjFrom(const ::nucleus::genomics::v1::GffHeader&, py::PostConv);
bool Clif_PyObjAs(PyObject*, std::unique_ptr<::nucleus::genomics::v1::GffHeader>*);
PyObject* Clif_PyObjFrom(std::unique_ptr<const ::nucleus::genomics::v1::GffHeader>, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<const ::nucleus::genomics::v1::GffHeader>, py::PostConv);
// CLIF use `::nucleus::genomics::v1::GffHeader::OntologyDirective` as GffHeader.OntologyDirective
bool Clif_PyObjAs(PyObject* input, ::nucleus::genomics::v1::GffHeader::OntologyDirective* output);
PyObject* Clif_PyObjFrom(const ::nucleus::genomics::v1::GffHeader::OntologyDirective&, py::PostConv);
bool Clif_PyObjAs(PyObject*, std::unique_ptr<::nucleus::genomics::v1::GffHeader::OntologyDirective>*);
PyObject* Clif_PyObjFrom(std::unique_ptr<const ::nucleus::genomics::v1::GffHeader::OntologyDirective>, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<const ::nucleus::genomics::v1::GffHeader::OntologyDirective>, py::PostConv);
// CLIF use `::nucleus::genomics::v1::GffHeader::GenomeBuildDirective` as GffHeader.GenomeBuildDirective
bool Clif_PyObjAs(PyObject* input, ::nucleus::genomics::v1::GffHeader::GenomeBuildDirective* output);
PyObject* Clif_PyObjFrom(const ::nucleus::genomics::v1::GffHeader::GenomeBuildDirective&, py::PostConv);
bool Clif_PyObjAs(PyObject*, std::unique_ptr<::nucleus::genomics::v1::GffHeader::GenomeBuildDirective>*);
PyObject* Clif_PyObjFrom(std::unique_ptr<const ::nucleus::genomics::v1::GffHeader::GenomeBuildDirective>, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<const ::nucleus::genomics::v1::GffHeader::GenomeBuildDirective>, py::PostConv);
// CLIF use `::nucleus::genomics::v1::GffReaderOptions` as GffReaderOptions
bool Clif_PyObjAs(PyObject* input, ::nucleus::genomics::v1::GffReaderOptions* output);
PyObject* Clif_PyObjFrom(const ::nucleus::genomics::v1::GffReaderOptions&, py::PostConv);
bool Clif_PyObjAs(PyObject*, std::unique_ptr<::nucleus::genomics::v1::GffReaderOptions>*);
PyObject* Clif_PyObjFrom(std::unique_ptr<const ::nucleus::genomics::v1::GffReaderOptions>, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<const ::nucleus::genomics::v1::GffReaderOptions>, py::PostConv);
// CLIF use `::nucleus::genomics::v1::GffWriterOptions` as GffWriterOptions
bool Clif_PyObjAs(PyObject* input, ::nucleus::genomics::v1::GffWriterOptions* output);
PyObject* Clif_PyObjFrom(const ::nucleus::genomics::v1::GffWriterOptions&, py::PostConv);
bool Clif_PyObjAs(PyObject*, std::unique_ptr<::nucleus::genomics::v1::GffWriterOptions>*);
PyObject* Clif_PyObjFrom(std::unique_ptr<const ::nucleus::genomics::v1::GffWriterOptions>, py::PostConv);
PyObject* Clif_PyObjFrom(std::shared_ptr<const ::nucleus::genomics::v1::GffWriterOptions>, py::PostConv);
// CLIF use `::nucleus::genomics::v1::GffRecord::Strand` as GffRecord.Strand
bool Clif_PyObjAs(PyObject* input, ::nucleus::genomics::v1::GffRecord::Strand* output);
PyObject* Clif_PyObjFrom(::nucleus::genomics::v1::GffRecord::Strand, py::PostConv);

} } }  // namespace nucleus::genomics::v1
