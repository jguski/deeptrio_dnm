// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: third_party/nucleus/protos/range.proto

#include "third_party/nucleus/protos/range.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
namespace nucleus {
namespace genomics {
namespace v1 {
class RangeDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<Range> _instance;
} _Range_default_instance_;
}  // namespace v1
}  // namespace genomics
}  // namespace nucleus
static void InitDefaultsscc_info_Range_third_5fparty_2fnucleus_2fprotos_2frange_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::nucleus::genomics::v1::_Range_default_instance_;
    new (ptr) ::nucleus::genomics::v1::Range();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::nucleus::genomics::v1::Range::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_Range_third_5fparty_2fnucleus_2fprotos_2frange_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsscc_info_Range_third_5fparty_2fnucleus_2fprotos_2frange_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_third_5fparty_2fnucleus_2fprotos_2frange_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_third_5fparty_2fnucleus_2fprotos_2frange_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_third_5fparty_2fnucleus_2fprotos_2frange_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_third_5fparty_2fnucleus_2fprotos_2frange_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::nucleus::genomics::v1::Range, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::nucleus::genomics::v1::Range, reference_name_),
  PROTOBUF_FIELD_OFFSET(::nucleus::genomics::v1::Range, start_),
  PROTOBUF_FIELD_OFFSET(::nucleus::genomics::v1::Range, end_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::nucleus::genomics::v1::Range)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::nucleus::genomics::v1::_Range_default_instance_),
};

const char descriptor_table_protodef_third_5fparty_2fnucleus_2fprotos_2frange_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n&third_party/nucleus/protos/range.proto"
  "\022\023nucleus.genomics.v1\";\n\005Range\022\026\n\016refere"
  "nce_name\030\001 \001(\t\022\r\n\005start\030\002 \001(\003\022\013\n\003end\030\003 \001"
  "(\003b\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_third_5fparty_2fnucleus_2fprotos_2frange_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_third_5fparty_2fnucleus_2fprotos_2frange_2eproto_sccs[1] = {
  &scc_info_Range_third_5fparty_2fnucleus_2fprotos_2frange_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_third_5fparty_2fnucleus_2fprotos_2frange_2eproto_once;
static bool descriptor_table_third_5fparty_2fnucleus_2fprotos_2frange_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_third_5fparty_2fnucleus_2fprotos_2frange_2eproto = {
  &descriptor_table_third_5fparty_2fnucleus_2fprotos_2frange_2eproto_initialized, descriptor_table_protodef_third_5fparty_2fnucleus_2fprotos_2frange_2eproto, "third_party/nucleus/protos/range.proto", 130,
  &descriptor_table_third_5fparty_2fnucleus_2fprotos_2frange_2eproto_once, descriptor_table_third_5fparty_2fnucleus_2fprotos_2frange_2eproto_sccs, descriptor_table_third_5fparty_2fnucleus_2fprotos_2frange_2eproto_deps, 1, 0,
  schemas, file_default_instances, TableStruct_third_5fparty_2fnucleus_2fprotos_2frange_2eproto::offsets,
  file_level_metadata_third_5fparty_2fnucleus_2fprotos_2frange_2eproto, 1, file_level_enum_descriptors_third_5fparty_2fnucleus_2fprotos_2frange_2eproto, file_level_service_descriptors_third_5fparty_2fnucleus_2fprotos_2frange_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_third_5fparty_2fnucleus_2fprotos_2frange_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_third_5fparty_2fnucleus_2fprotos_2frange_2eproto), true);
namespace nucleus {
namespace genomics {
namespace v1 {

// ===================================================================

void Range::InitAsDefaultInstance() {
}
class Range::_Internal {
 public:
};

Range::Range()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:nucleus.genomics.v1.Range)
}
Range::Range(const Range& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  reference_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from.reference_name().empty()) {
    reference_name_.AssignWithDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from.reference_name_);
  }
  ::memcpy(&start_, &from.start_,
    static_cast<size_t>(reinterpret_cast<char*>(&end_) -
    reinterpret_cast<char*>(&start_)) + sizeof(end_));
  // @@protoc_insertion_point(copy_constructor:nucleus.genomics.v1.Range)
}

void Range::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_Range_third_5fparty_2fnucleus_2fprotos_2frange_2eproto.base);
  reference_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  ::memset(&start_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&end_) -
      reinterpret_cast<char*>(&start_)) + sizeof(end_));
}

Range::~Range() {
  // @@protoc_insertion_point(destructor:nucleus.genomics.v1.Range)
  SharedDtor();
}

void Range::SharedDtor() {
  reference_name_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void Range::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const Range& Range::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_Range_third_5fparty_2fnucleus_2fprotos_2frange_2eproto.base);
  return *internal_default_instance();
}


void Range::Clear() {
// @@protoc_insertion_point(message_clear_start:nucleus.genomics.v1.Range)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  reference_name_.ClearToEmptyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  ::memset(&start_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&end_) -
      reinterpret_cast<char*>(&start_)) + sizeof(end_));
  _internal_metadata_.Clear();
}

#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
const char* Range::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // string reference_name = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParserUTF8(mutable_reference_name(), ptr, ctx, "nucleus.genomics.v1.Range.reference_name");
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // int64 start = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          start_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // int64 end = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          end_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}
#else  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
bool Range::MergePartialFromCodedStream(
    ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!PROTOBUF_PREDICT_TRUE(EXPRESSION)) goto failure
  ::PROTOBUF_NAMESPACE_ID::uint32 tag;
  // @@protoc_insertion_point(parse_start:nucleus.genomics.v1.Range)
  for (;;) {
    ::std::pair<::PROTOBUF_NAMESPACE_ID::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // string reference_name = 1;
      case 1: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (10 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadString(
                input, this->mutable_reference_name()));
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
            this->reference_name().data(), static_cast<int>(this->reference_name().length()),
            ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE,
            "nucleus.genomics.v1.Range.reference_name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int64 start = 2;
      case 2: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (16 & 0xFF)) {

          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   ::PROTOBUF_NAMESPACE_ID::int64, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT64>(
                 input, &start_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int64 end = 3;
      case 3: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (24 & 0xFF)) {

          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   ::PROTOBUF_NAMESPACE_ID::int64, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT64>(
                 input, &end_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:nucleus.genomics.v1.Range)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:nucleus.genomics.v1.Range)
  return false;
#undef DO_
}
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER

void Range::SerializeWithCachedSizes(
    ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:nucleus.genomics.v1.Range)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string reference_name = 1;
  if (this->reference_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->reference_name().data(), static_cast<int>(this->reference_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "nucleus.genomics.v1.Range.reference_name");
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->reference_name(), output);
  }

  // int64 start = 2;
  if (this->start() != 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt64(2, this->start(), output);
  }

  // int64 end = 3;
  if (this->end() != 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt64(3, this->end(), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFields(
        _internal_metadata_.unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:nucleus.genomics.v1.Range)
}

::PROTOBUF_NAMESPACE_ID::uint8* Range::InternalSerializeWithCachedSizesToArray(
    ::PROTOBUF_NAMESPACE_ID::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:nucleus.genomics.v1.Range)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string reference_name = 1;
  if (this->reference_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->reference_name().data(), static_cast<int>(this->reference_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "nucleus.genomics.v1.Range.reference_name");
    target =
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringToArray(
        1, this->reference_name(), target);
  }

  // int64 start = 2;
  if (this->start() != 0) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt64ToArray(2, this->start(), target);
  }

  // int64 end = 3;
  if (this->end() != 0) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt64ToArray(3, this->end(), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:nucleus.genomics.v1.Range)
  return target;
}

size_t Range::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:nucleus.genomics.v1.Range)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::ComputeUnknownFieldsSize(
        _internal_metadata_.unknown_fields());
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string reference_name = 1;
  if (this->reference_name().size() > 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->reference_name());
  }

  // int64 start = 2;
  if (this->start() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int64Size(
        this->start());
  }

  // int64 end = 3;
  if (this->end() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int64Size(
        this->end());
  }

  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Range::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:nucleus.genomics.v1.Range)
  GOOGLE_DCHECK_NE(&from, this);
  const Range* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<Range>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:nucleus.genomics.v1.Range)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:nucleus.genomics.v1.Range)
    MergeFrom(*source);
  }
}

void Range::MergeFrom(const Range& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:nucleus.genomics.v1.Range)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.reference_name().size() > 0) {

    reference_name_.AssignWithDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from.reference_name_);
  }
  if (from.start() != 0) {
    set_start(from.start());
  }
  if (from.end() != 0) {
    set_end(from.end());
  }
}

void Range::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:nucleus.genomics.v1.Range)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Range::CopyFrom(const Range& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:nucleus.genomics.v1.Range)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Range::IsInitialized() const {
  return true;
}

void Range::InternalSwap(Range* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  reference_name_.Swap(&other->reference_name_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(start_, other->start_);
  swap(end_, other->end_);
}

::PROTOBUF_NAMESPACE_ID::Metadata Range::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace v1
}  // namespace genomics
}  // namespace nucleus
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::nucleus::genomics::v1::Range* Arena::CreateMaybeMessage< ::nucleus::genomics::v1::Range >(Arena* arena) {
  return Arena::CreateInternal< ::nucleus::genomics::v1::Range >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
